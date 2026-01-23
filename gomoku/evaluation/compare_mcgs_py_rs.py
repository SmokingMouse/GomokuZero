import argparse
import importlib
import random
import sys
from pathlib import Path

import numpy as np
import torch

from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcgs import ZeroMCGS


def parse_moves(moves_text: str) -> list[int]:
    if not moves_text:
        return []
    return [int(token) for token in moves_text.split(",") if token.strip()]


def make_policy_fn(policy: ZeroPolicy, device: str):
    policy.eval()

    def policy_fn(obs: np.ndarray):
        x = torch.from_numpy(obs).reshape(1, -1).float().to(device)
        with torch.no_grad():
            policy_logits, value = policy(x)
        probs = policy_logits.squeeze(0).cpu().numpy()
        return probs, float(value.item())

    return policy_fn


def build_env(board_size: int, moves: list[int]) -> GomokuEnv:
    env = GomokuEnv(board_size=board_size)
    env.reset()
    for action in moves:
        env.step(action)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Python ZeroMCGS vs Rust ZeroMCGS with fixed state/seed."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--moves",
        type=str,
        default="40,41,31,32,22,23",
        help="Comma-separated action list applied before search.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
    zero_mcts_rs = importlib.import_module("zero_mcts_rs")
    if not hasattr(zero_mcts_rs, "ZeroMCGS"):
        raise RuntimeError(
            "zero_mcts_rs module loaded, but ZeroMCGS is missing. "
            "Run maturin develop in zero_mcts_rs first."
        )

    policy = ZeroPolicy(
        board_size=args.board_size,
        num_blocks=args.num_blocks,
        base_channels=args.base_channels,
    ).to(args.device)
    policy.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    policy.eval()

    policy_fn = make_policy_fn(policy, args.device)
    env = build_env(args.board_size, parse_moves(args.moves))

    py_mcts = ZeroMCGS(
        policy,
        puct=2.0,
        device=args.device,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    py_action, py_pi = py_mcts.run(
        env, iterations=args.iterations, temperature=args.temperature
    )

    rs_mcts = zero_mcts_rs.ZeroMCGS(
        board_size=args.board_size,
        puct=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    board_flat = env.board.astype(np.int8).reshape(-1)
    rs_action, rs_pi = rs_mcts.run(
        board_flat,
        env.current_player,
        policy_fn,
        iterations=args.iterations,
        temperature=args.temperature,
        use_dirichlet=True,
        move_size=env.move_size,
        last_action=env.last_action,
    )

    rs_pi = np.asarray(rs_pi)
    py_pi = np.asarray(py_pi)

    pi_diff = np.max(np.abs(rs_pi - py_pi))
    print(f"py_action={py_action}, rs_action={rs_action}")
    print(f"pi_max_abs_diff={pi_diff:.6f}")
    if py_action != rs_action or pi_diff > 1e-6:
        top_py = int(np.argmax(py_pi))
        top_rs = int(np.argmax(rs_pi))
        print(f"top_py={top_py}, top_rs={top_rs}")


if __name__ == "__main__":
    main()

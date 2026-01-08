import importlib
import sys
import time
from pathlib import Path

import numpy as np
import torch

from gomoku.gomoku_env import GomokuEnv
from gomoku.light_zero_mcts import LightZeroMCTS
from gomoku.policy import ZeroPolicy


def make_policy_fn(policy: ZeroPolicy, device: str):
    policy.eval()

    def policy_fn(obs: np.ndarray):
        x = torch.from_numpy(obs).reshape(1, -1).float().to(device)
        with torch.no_grad():
            policy_logits, value = policy(x)
        probs = policy_logits.squeeze(0).cpu().numpy()
        return probs, float(value.item())

    return policy_fn


def play_match(
    env: GomokuEnv,
    rs_mcts,
    py_mcts: LightZeroMCTS,
    policy_fn,
    iterations: int,
    temperature: float,
    render_steps: bool,
    rs_time: list[float],
    py_time: list[float],
) -> None:
    env.reset()
    if render_steps:
        env.render()
    while not env.done:
        current_player = env.current_player
        if current_player == 1:
            board_flat = env.board.astype(np.int8).reshape(-1)
            start = time.perf_counter()
            rs_mcts.run(
                board_flat,
                env.current_player,
                policy_fn,
                iterations=iterations,
                use_dirichlet=False,
                move_size=env.move_size,
                last_action=env.last_action,
            )
            rs_time[0] += time.perf_counter() - start
            action, _ = rs_mcts.select_action_with_temperature(
                temperature=temperature, top_k=None
            )
        else:
            start = time.perf_counter()
            py_mcts.run(env, iterations=iterations, use_dirichlet=False)
            py_time[0] += time.perf_counter() - start
            action, _ = py_mcts.select_action_with_temperature(
                temperature=temperature, top_k=None
            )

        env.step(action)
        rs_mcts.step(action)
        py_mcts.step(action)
        if render_steps:
            print(f"move {env.move_size}: player={current_player}, action={action}")
            env.render()


def main():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) in sys.path:
        sys.path.remove(str(repo_root))
    zero_mcts_rs = importlib.import_module("zero_mcts_rs")
    if not hasattr(zero_mcts_rs, "LightZeroMCTS"):
        raise RuntimeError(
            "zero_mcts_rs module loaded, but LightZeroMCTS is missing. "
            "Run maturin develop in zero_mcts_rs first."
        )

    board_size = 9
    iterations = 200
    temperature = 0.2
    num_games = 2
    render_steps = True

    ckpt_path = repo_root / "gomoku" / "continue_model" / "policy_step_940000.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    device = "cpu"
    policy = ZeroPolicy(board_size=9, num_blocks=2).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))

    policy_fn = make_policy_fn(policy, device)

    rs_wins = 0
    py_wins = 0
    draws = 0
    rs_time = [0.0]
    py_time = [0.0]
    start = time.perf_counter()

    for game_idx in range(num_games):
        env = GomokuEnv(board_size=board_size)

        rs_mcts = zero_mcts_rs.LightZeroMCTS(
            board_size=board_size,
            puct=2.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )
        py_mcts = LightZeroMCTS(
            policy,
            puct=2.0,
            device=device,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )

        if game_idx % 2 == 1:
            env.current_player = 2

        if render_steps:
            print(f"=== game {game_idx + 1} ===")

        play_match(
            env,
            rs_mcts,
            py_mcts,
            policy_fn,
            iterations=iterations,
            temperature=temperature,
            render_steps=render_steps,
            rs_time=rs_time,
            py_time=py_time,
        )

        winner = env.winner
        if winner is None or winner == 0:
            draws += 1
        else:
            rs_is_player1 = game_idx % 2 == 0
            rs_is_winner = (winner == 1 and rs_is_player1) or (
                winner == 2 and not rs_is_player1
            )
            if rs_is_winner:
                rs_wins += 1
            else:
                py_wins += 1

    elapsed = time.perf_counter() - start
    total = rs_wins + py_wins + draws
    print(
        f"rs vs py ({total} games): rs_wins={rs_wins}, "
        f"py_wins={py_wins}, draws={draws}, elapsed={elapsed:.3f}s, "
        f"rs_time={rs_time[0]:.3f}s, py_time={py_time[0]:.3f}s"
    )


if __name__ == "__main__":
    main()

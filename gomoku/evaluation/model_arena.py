import argparse
import time
from pathlib import Path

import torch

from gomoku.gomoku_env import GomokuEnv
from gomoku.light_zero_mcts import LightZeroMCTS
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcgs import ZeroMCGS


def load_policy(
    ckpt_path: Path,
    board_size: int,
    num_blocks: int,
    base_channels: int,
    device: str,
) -> ZeroPolicy:
    policy = ZeroPolicy(
        board_size=board_size,
        num_blocks=num_blocks,
        base_channels=base_channels,
    ).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()
    return policy


def make_mcts(
    engine: str,
    policy: ZeroPolicy,
    device: str,
    puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
):
    if engine == "zero_mcgs":
        return ZeroMCGS(
            policy,
            puct=puct,
            device=device,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )
    if engine == "light_zero_mcts":
        return LightZeroMCTS(
            policy,
            puct=puct,
            device=device,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
        )
    raise ValueError(f"unknown engine: {engine}")


def select_action(
    engine: str,
    mcts,
    env: GomokuEnv,
    iterations: int,
    temperature: float,
    use_dirichlet: bool,
) -> int:
    if engine == "zero_mcgs":
        action, _ = mcts.run(
            env,
            iterations=iterations,
            temperature=temperature,
        )
        return action

    mcts.run(env, iterations=iterations, use_dirichlet=use_dirichlet)
    action, _ = mcts.select_action_with_temperature(temperature)
    return action


def play_game(
    engine: str,
    policy_a: ZeroPolicy,
    policy_b: ZeroPolicy,
    board_size: int,
    iterations: int,
    temperature: float,
    puct: float,
    dirichlet_alpha: float,
    dirichlet_epsilon: float,
    a_is_player1: bool,
    use_dirichlet: bool,
    render_steps: bool,
    device: str,
) -> int:
    env = GomokuEnv(board_size=board_size)
    if not a_is_player1:
        env.current_player = 2

    mcts_a = make_mcts(
        engine, policy_a, device, puct, dirichlet_alpha, dirichlet_epsilon
    )
    mcts_b = make_mcts(
        engine, policy_b, device, puct, dirichlet_alpha, dirichlet_epsilon
    )

    if render_steps:
        env.render()

    while not env.done:
        current_player = env.current_player
        if current_player == 1:
            mcts = mcts_a if a_is_player1 else mcts_b
        else:
            mcts = mcts_b if a_is_player1 else mcts_a

        action = select_action(
            engine,
            mcts,
            env,
            iterations=iterations,
            temperature=temperature,
            use_dirichlet=use_dirichlet,
        )
        env.step(action)
        mcts_a.step(action)
        mcts_b.step(action)
        if render_steps:
            print(f"move {env.move_size}: player={current_player}, action={action}")
            env.render()

    return env.winner or 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Arena matches between two models.")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--num-blocks", type=int, default=2)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument(
        "--engine",
        type=str,
        default="zero_mcgs",
        choices=("zero_mcgs", "light_zero_mcts"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--puct", type=float, default=2.0)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-epsilon", type=float, default=0.25)
    parser.add_argument("--use-dirichlet", action="store_true")
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    policy_a = load_policy(
        args.model_a,
        board_size=args.board_size,
        num_blocks=args.num_blocks,
        base_channels=args.base_channels,
        device=args.device,
    )
    policy_b = load_policy(
        args.model_b,
        board_size=args.board_size,
        num_blocks=args.num_blocks,
        base_channels=args.base_channels,
        device=args.device,
    )

    a_wins = 0
    b_wins = 0
    draws = 0
    start = time.perf_counter()

    for game_idx in range(args.games):
        a_is_player1 = game_idx % 2 == 0
        winner = play_game(
            engine=args.engine,
            policy_a=policy_a,
            policy_b=policy_b,
            board_size=args.board_size,
            iterations=args.iterations,
            temperature=args.temperature,
            puct=args.puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_epsilon=args.dirichlet_epsilon,
            a_is_player1=a_is_player1,
            use_dirichlet=args.use_dirichlet,
            render_steps=args.render,
            device=args.device,
        )

        if winner == 0:
            draws += 1
        else:
            a_won = (winner == 1 and a_is_player1) or (winner == 2 and not a_is_player1)
            if a_won:
                a_wins += 1
            else:
                b_wins += 1

    elapsed = time.perf_counter() - start
    total = a_wins + b_wins + draws
    print(
        f"arena ({total} games): a_wins={a_wins}, b_wins={b_wins}, "
        f"draws={draws}, elapsed={elapsed:.3f}s"
    )


if __name__ == "__main__":
    main()

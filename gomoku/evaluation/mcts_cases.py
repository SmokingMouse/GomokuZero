import random
from typing import Tuple

import numpy as np
import torch

from gomoku.gomoku_env import GomokuEnv
from gomoku.light_zero_mcts import LightZeroMCTS
from gomoku.worker import get_symmetric_data


class DummyPolicy(torch.nn.Module):
    def __init__(
        self, board_size: int, bias_action: int | None = None, bias_logit: float = 5.0
    ):
        super().__init__()
        self.board_size = board_size
        self.bias_action = bias_action
        self.bias_logit = bias_logit

    def forward(self, x):
        batch = x.shape[0]
        logits = torch.zeros(
            (batch, self.board_size * self.board_size),
            dtype=torch.float32,
            device=x.device,
        )
        if self.bias_action is not None:
            logits[:, self.bias_action] = self.bias_logit
        value = torch.zeros((batch, 1), dtype=torch.float32, device=x.device)
        return logits, value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def make_policy_fn(
    board_size: int, bias_action: int | None = None, bias_logit: float = 5.0
):
    def policy_fn(obs: np.ndarray):
        _ = np.asarray(obs)
        logits = np.zeros((board_size * board_size,), dtype=np.float32)
        if bias_action is not None:
            logits[bias_action] = bias_logit
        value = 0.0
        return logits, value

    return policy_fn


def make_env_with_board(
    board: np.ndarray, current_player: int, board_size: int
) -> GomokuEnv:
    env = GomokuEnv(board_size)
    env.board = board.copy()
    env.current_player = current_player
    env.done = False
    env.winner = None
    env.last_action = -1
    env.move_size = int(np.count_nonzero(board))
    return env


def case_single_valid_move() -> Tuple[bool, str]:
    board_size = 3
    board = np.ones((board_size, board_size), dtype=np.int8)
    board[1, 1] = 0
    env = make_env_with_board(board, current_player=1, board_size=board_size)
    policy = DummyPolicy(board_size=board_size)
    mcts = LightZeroMCTS(policy, device="cpu")
    action, _ = mcts.run(env, iterations=40)
    expected_action = 1 * board_size + 1
    passed = action == expected_action
    msg = f"single_valid_move: action={action}, expected={expected_action}"
    return passed, msg


def case_immediate_win() -> Tuple[bool, str]:
    board_size = 9
    board = np.zeros((board_size, board_size), dtype=np.int8)
    row = 4
    for col in range(4):
        board[row, col] = 1
    env = make_env_with_board(board, current_player=1, board_size=board_size)
    expected_action = row * board_size + 4
    policy = DummyPolicy(board_size=board_size, bias_action=expected_action)
    mcts = LightZeroMCTS(policy, device="cpu")
    action, _ = mcts.run(env, iterations=200)
    expected_action = row * board_size + 4
    passed = action == expected_action
    msg = f"immediate_win: action={action}, expected={expected_action}"
    return passed, msg


def case_hash_collision() -> Tuple[bool, str]:
    board_size = 5
    board = np.zeros((board_size, board_size), dtype=np.int8)
    board[2, 2] = 1
    env_p1 = make_env_with_board(board, current_player=1, board_size=board_size)
    env_p2 = make_env_with_board(board, current_player=2, board_size=board_size)
    policy = DummyPolicy(board_size=board_size)
    mcts = LightZeroMCTS(policy, device="cpu")
    mcts.run(env_p1, iterations=2)
    root_id_p1 = id(mcts.root)
    mcts.run(env_p2, iterations=2)
    root_id_p2 = id(mcts.root)
    passed = root_id_p1 != root_id_p2
    msg = f"hash_collision: root_id_p1={root_id_p1}, root_id_p2={root_id_p2}"
    return passed, msg


def case_symmetry_flip() -> Tuple[bool, str]:
    board_size = 5
    state = np.zeros((3, board_size, board_size), dtype=np.int8)
    state[0, 1, 0] = 1
    pi = [0.0] * (board_size * board_size)
    pi[1 * board_size + 0] = 1.0

    samples = get_symmetric_data(state, pi, board_size=board_size)
    rotated_state, _ = samples[1]
    expected = np.flip(np.rot90(state, axes=(1, 2), k=0), axis=2)
    passed = np.array_equal(rotated_state, expected)
    msg = "symmetry_flip: state flip matches expected axis"
    return passed, msg


def case_rs_single_valid_move() -> Tuple[bool, str]:
    try:
        import importlib

        zero_mcts_rs = importlib.import_module("zero_mcts_rs")
    except Exception as exc:
        return False, f"rs_single_valid_move: import failed ({exc})"

    board_size = 3
    board = np.ones((board_size, board_size), dtype=np.int8)
    board[1, 1] = 0
    env = make_env_with_board(board, current_player=1, board_size=board_size)
    rs_mcts = zero_mcts_rs.LightZeroMCTS(
        board_size=board_size,
        puct=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    policy_fn = make_policy_fn(board_size=board_size)
    board_flat = env.board.astype(np.int8).reshape(-1)
    rs_mcts.run(
        board_flat,
        env.current_player,
        policy_fn,
        iterations=40,
        use_dirichlet=False,
        move_size=env.move_size,
        last_action=env.last_action,
    )
    action, _ = rs_mcts.select_action_with_temperature(temperature=0, top_k=None)
    expected_action = 1 * board_size + 1
    passed = action == expected_action
    msg = f"rs_single_valid_move: action={action}, expected={expected_action}"
    return passed, msg


def case_rs_immediate_win() -> Tuple[bool, str]:
    try:
        import importlib

        zero_mcts_rs = importlib.import_module("zero_mcts_rs")
    except Exception as exc:
        return False, f"rs_immediate_win: import failed ({exc})"

    board_size = 9
    board = np.zeros((board_size, board_size), dtype=np.int8)
    row = 4
    for col in range(4):
        board[row, col] = 1
    env = make_env_with_board(board, current_player=1, board_size=board_size)
    expected_action = row * board_size + 4
    rs_mcts = zero_mcts_rs.LightZeroMCTS(
        board_size=board_size,
        puct=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    policy_fn = make_policy_fn(
        board_size=board_size, bias_action=expected_action, bias_logit=12.0
    )
    board_flat = env.board.astype(np.int8).reshape(-1)
    rs_mcts.run(
        board_flat,
        env.current_player,
        policy_fn,
        iterations=200,
        use_dirichlet=False,
        move_size=env.move_size,
        last_action=env.last_action,
    )
    action, _ = rs_mcts.select_action_with_temperature(temperature=0, top_k=None)
    passed = action == expected_action
    msg = f"rs_immediate_win: action={action}, expected={expected_action}"
    return passed, msg


def case_rs_vs_py_biased_action() -> Tuple[bool, str]:
    try:
        import importlib

        zero_mcts_rs = importlib.import_module("zero_mcts_rs")
    except Exception as exc:
        return False, f"rs_vs_py_biased_action: import failed ({exc})"

    board_size = 5
    bias_action = 0
    board = np.zeros((board_size, board_size), dtype=np.int8)
    env = make_env_with_board(board, current_player=1, board_size=board_size)

    py_policy = DummyPolicy(
        board_size=board_size, bias_action=bias_action, bias_logit=12.0
    )
    py_mcts = LightZeroMCTS(py_policy, device="cpu")
    py_action, _ = py_mcts.run(env, iterations=100, use_dirichlet=False)

    rs_mcts = zero_mcts_rs.LightZeroMCTS(
        board_size=board_size,
        puct=2.0,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
    )
    policy_fn = make_policy_fn(
        board_size=board_size, bias_action=bias_action, bias_logit=12.0
    )
    board_flat = env.board.astype(np.int8).reshape(-1)
    rs_mcts.run(
        board_flat,
        env.current_player,
        policy_fn,
        iterations=100,
        use_dirichlet=False,
        move_size=env.move_size,
        last_action=env.last_action,
    )
    rs_action, _ = rs_mcts.select_action_with_temperature(temperature=0, top_k=None)

    passed = py_action == rs_action == bias_action
    msg = (
        "rs_vs_py_biased_action: "
        f"py_action={py_action}, rs_action={rs_action}, expected={bias_action}"
    )
    return passed, msg


def run_cases() -> None:
    seed_everything(7)
    cases = [
        case_single_valid_move,
        case_immediate_win,
        case_hash_collision,
        case_symmetry_flip,
        case_rs_single_valid_move,
        case_rs_immediate_win,
        case_rs_vs_py_biased_action,
    ]
    for case in cases:
        passed, msg = case()
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {msg}")


if __name__ == "__main__":
    run_cases()

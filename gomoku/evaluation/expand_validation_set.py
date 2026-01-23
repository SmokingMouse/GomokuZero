import argparse
import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def expand_board(board, src_size, dst_size, offset):
    if offset is None:
        offset = (dst_size - src_size) // 2
    dst = [[0 for _ in range(dst_size)] for _ in range(dst_size)]
    for r in range(src_size):
        for c in range(src_size):
            dst[r + offset][c + offset] = board[r][c]
    return dst


def expand_planes(planes, src_size, dst_size, offset):
    if offset is None:
        offset = (dst_size - src_size) // 2
    expanded = []
    for plane in planes:
        expanded.append(expand_board(plane, src_size, dst_size, offset))
    return expanded


def remap_action(action, src_size, dst_size, offset):
    if offset is None:
        offset = (dst_size - src_size) // 2
    if action is None or action < 0:
        return action
    row = action // src_size
    col = action % src_size
    return (row + offset) * dst_size + (col + offset)


def normalize_best_actions(best_action):
    if isinstance(best_action, list):
        return [int(a) for a in best_action if isinstance(a, int)]
    if isinstance(best_action, int):
        return [best_action]
    return []


def expand_state(state, src_size, dst_size, offset):
    if isinstance(state, dict) and "board" in state:
        next_state = copy.deepcopy(state)
        board = state["board"]
        if (
            isinstance(board, list)
            and board
            and isinstance(board[0], list)
            and len(board) == src_size
            and len(board[0]) == src_size
        ):
            next_state["board"] = expand_board(board, src_size, dst_size, offset)
        elif (
            isinstance(board, list)
            and len(board) in (2, 3)
            and isinstance(board[0], list)
            and len(board[0]) == src_size
        ):
            next_state["board"] = expand_planes(board, src_size, dst_size, offset)
        else:
            return None
        last_action = state.get("last_action", None)
        if isinstance(last_action, int):
            next_state["last_action"] = remap_action(
                last_action, src_size, dst_size, offset
            )
        return next_state

    if isinstance(state, list) and len(state) in (2, 3):
        if isinstance(state[0], list) and len(state[0]) == src_size:
            return expand_planes(state, src_size, dst_size, offset)
    if (
        isinstance(state, list)
        and state
        and isinstance(state[0], list)
        and len(state) == src_size
        and len(state[0]) == src_size
    ):
        return expand_board(state, src_size, dst_size, offset)
    return None


def expand_entry(entry, src_size, dst_size, offset):
    next_entry = copy.deepcopy(entry)
    next_state = expand_state(entry.get("state"), src_size, dst_size, offset)
    if next_state is None:
        return None
    next_entry["state"] = next_state
    next_entry["board_size"] = dst_size
    best_actions = normalize_best_actions(entry.get("best_action"))
    if best_actions:
        next_entry["best_action"] = [
            remap_action(action, src_size, dst_size, offset) for action in best_actions
        ]
    next_entry["id"] = uuid.uuid4().hex
    next_entry["created_at"] = datetime.now(timezone.utc).isoformat()
    return next_entry


def expand_file(input_path, output_path, src_size, dst_size, offset):
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as handle:
        lines = [line for line in handle if line.strip()]

    expanded = []
    for line in lines:
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("board_size") != src_size:
            continue
        next_entry = expand_entry(entry, src_size, dst_size, offset)
        if next_entry is not None:
            expanded.append(next_entry)

    with output_path.open("w", encoding="utf-8") as handle:
        for entry in expanded:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return len(expanded)


def main():
    parser = argparse.ArgumentParser(
        description="Expand Gomoku validation set boards to a larger size."
    )
    parser.add_argument(
        "--input",
        default="gomoku/validation_sets/validation_set.jsonl",
        help="Input validation jsonl path.",
    )
    parser.add_argument(
        "--output",
        default="gomoku/validation_sets/validation_set_15.jsonl",
        help="Output validation jsonl path.",
    )
    parser.add_argument("--src-size", type=int, default=9)
    parser.add_argument("--dst-size", type=int, default=15)
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
        help="Top-left offset; defaults to centering.",
    )
    args = parser.parse_args()
    count = expand_file(args.input, args.output, args.src_size, args.dst_size, args.offset)
    print(f"Expanded {count} entries into {args.output}")


if __name__ == "__main__":
    main()

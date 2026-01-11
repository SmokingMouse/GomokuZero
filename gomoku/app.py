from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from gomoku.gomoku_env import GomokuEnvSimple
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcts import ZeroMCTS

BOARD_SIZE = 9
DEFAULT_MODEL_PATH = os.getenv(
    "GOMOKU_MODEL_PATH",
    # "models/continue_model/policy_step_10000.pth",
    # "models/gomoku_zero_9_pre5/policy_step_990000.pth",
    # "models/gomoku_zero_9_lab_2/policy_step_900000.pth",
    # "models/gomoku_zero_9_lab_3/policy_step_250000.pth",
    "models/gomoku_zero_9_lab_4/policy_step_30000.pth",
)
MCTS_ITERATIONS = int(os.getenv("GOMOKU_MCTS_ITERS", "400"))
MCTS_PUCT = float(os.getenv("GOMOKU_MCTS_PUCT", "2.0"))
MAX_WORKERS = int(os.getenv("GOMOKU_AI_WORKERS", "2"))
VALIDATION_DIR = Path(__file__).resolve().parent / "validation_sets"
VALIDATION_FILE = VALIDATION_DIR / "validation_set.jsonl"


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / path


def load_policy(model_path: str) -> ZeroPolicy:
    policy = ZeroPolicy(board_size=BOARD_SIZE)
    resolved_path = resolve_model_path(model_path)
    state = torch.load(resolved_path, map_location="cpu")
    policy.load_state_dict(state)
    policy.eval()
    return policy


POLICY = load_policy(DEFAULT_MODEL_PATH)
AI_POOL = ThreadPoolExecutor(max_workers=MAX_WORKERS)


@dataclass
class GameSession:
    env: GomokuEnvSimple
    human_player: int
    mode: str = "human_vs_ai"
    ai_task: Optional[asyncio.Task] = None
    ai_paused: bool = False
    ai_stop: bool = False
    last_move: Optional[dict[str, Any]] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.sessions: dict[WebSocket, GameSession] = {}

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        session = self.sessions.pop(websocket, None)
        if session is not None:
            cancel_ai_task(session)

    async def send_json(self, message: dict[str, Any], websocket: WebSocket):
        await websocket.send_json(message)

    def new_session(
        self,
        websocket: WebSocket,
        human_player: int,
        mode: str = "human_vs_ai",
        preset: Optional[dict[str, Any]] = None,
    ) -> GameSession:
        env = GomokuEnvSimple(board_size=BOARD_SIZE)
        env.reset()
        session = GameSession(env=env, human_player=human_player, mode=mode)
        if preset is not None:
            apply_preset_state(session, preset)
        self.sessions[websocket] = session
        return session


class ValidationSample(BaseModel):
    state: Any
    best_action: Union[int, list[int]] = Field(...)
    board_size: int = Field(default=BOARD_SIZE, ge=1)
    meta: Optional[dict[str, Any]] = None


class ValidationUpdate(BaseModel):
    id: str
    state: Optional[Any] = None
    best_action: Optional[Union[int, list[int]]] = None
    board_size: Optional[int] = Field(default=None, ge=1)
    meta: Optional[dict[str, Any]] = None


class ValidationDelete(BaseModel):
    id: str


def is_board_2d(state: Any, board_size: int) -> bool:
    if not isinstance(state, list) or len(state) != board_size:
        return False
    for row in state:
        if not isinstance(row, list) or len(row) != board_size:
            return False
    return True


def is_state_3d(state: Any, board_size: int) -> bool:
    if not isinstance(state, list):
        return False
    if len(state) not in (2, 3):
        return False
    return all(is_board_2d(channel, board_size) for channel in state)


def validate_state_payload(state: Any, board_size: int) -> None:
    if isinstance(state, dict):
        board = state.get("board")
        if not is_board_2d(board, board_size):
            raise ValueError("state.board must be board_size x board_size")
        return
    if is_board_2d(state, board_size):
        return
    if is_state_3d(state, board_size):
        return
    raise ValueError(
        "state must be board_size x board_size or channels x board_size x board_size"
    )


def normalize_best_action(best_action: Union[int, list[int]]) -> list[int]:
    if isinstance(best_action, int):
        return [best_action]
    if isinstance(best_action, list) and all(isinstance(a, int) for a in best_action):
        return best_action
    raise ValueError("best_action must be int or list[int]")


def normalize_state_payload(state: Any, board_size: int) -> dict[str, Any]:
    if isinstance(state, dict):
        board = state.get("board")
        current_player = state.get("current_player")
        last_action = state.get("last_action", -1)
        return {
            "board": board,
            "current_player": current_player,
            "last_action": last_action,
        }
    return {"board": state, "current_player": None, "last_action": -1}


def read_validation_entries() -> list[dict[str, Any]]:
    if not VALIDATION_FILE.exists():
        return []
    entries = []
    with VALIDATION_FILE.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def write_validation_entries(entries: list[dict[str, Any]]) -> None:
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with VALIDATION_FILE.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


app = FastAPI()
manager = ConnectionManager()

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("GOMOKU_CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def game_state_payload(session: GameSession) -> dict[str, Any]:
    winner = session.env.winner if session.env.winner is not None else 0
    return {
        "type": "game_state",
        "board": session.env.board.tolist(),
        "current_player": session.env.current_player,
        "winner": winner,
        "done": session.env.done,
        "last_move": session.last_move,
        "mode": session.mode,
    }


def compute_ai_action(env: GomokuEnvSimple) -> int:
    zero_mcts = ZeroMCTS(
        policy=POLICY,
        puct=MCTS_PUCT,
        device="cpu",
        dirichlet_alpha=0.0,
        dirichlet_epsilon=0.0,
    )
    action, _ = zero_mcts.run(env, iterations=MCTS_ITERATIONS, use_dirichlet=False)
    return int(action)


async def apply_ai_move(session: GameSession) -> None:
    loop = asyncio.get_running_loop()
    env_snapshot = session.env.clone()
    action = await loop.run_in_executor(AI_POOL, compute_ai_action, env_snapshot)
    row, col = action // BOARD_SIZE, action % BOARD_SIZE
    move_player = session.env.current_player
    session.env.step(action)
    session.last_move = {"x": row, "y": col, "player": move_player}


async def run_ai_vs_ai(
    session: GameSession, websocket: WebSocket, delay_ms: int = 0
) -> None:
    try:
        while not session.env.done and not session.ai_stop:
            if session.ai_paused:
                await asyncio.sleep(0.1)
                continue
            await apply_ai_move(session)
            await manager.send_json(game_state_payload(session), websocket)
            if delay_ms > 0:
                await asyncio.sleep(delay_ms / 1000.0)
    except Exception:
        session.ai_stop = True


def parse_human_player(payload: dict[str, Any]) -> int:
    human_player = int(payload.get("human_player", 2))
    if human_player not in (1, 2):
        raise ValueError("human_player must be 1 or 2")
    return human_player


def parse_preset(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    preset = payload.get("preset")
    if preset is None:
        return None
    if not isinstance(preset, dict):
        raise ValueError("preset must be an object")
    board = preset.get("board")
    if not isinstance(board, list) or len(board) != BOARD_SIZE:
        raise ValueError("preset.board must match board_size")
    for row in board:
        if not isinstance(row, list) or len(row) != BOARD_SIZE:
            raise ValueError("preset.board must be a square board")
        for cell in row:
            if cell not in (0, 1, 2):
                raise ValueError("preset.board values must be 0, 1, or 2")
    current_player = preset.get("current_player")
    if current_player is not None and current_player not in (1, 2):
        raise ValueError("preset.current_player must be 1 or 2")
    last_action = preset.get("last_action", -1)
    if last_action is not None and not isinstance(last_action, int):
        raise ValueError("preset.last_action must be int")
    return {
        "board": board,
        "current_player": current_player,
        "last_action": last_action,
    }


def infer_current_player_from_board(board: list[list[int]]) -> int:
    black = sum(cell == 1 for row in board for cell in row)
    white = sum(cell == 2 for row in board for cell in row)
    if black == white:
        return 1
    if black == white + 1:
        return 2
    return 1 if black <= white else 2


def apply_preset_state(session: GameSession, preset: dict[str, Any]) -> None:
    board = preset["board"]
    session.env.board = np.array(board, dtype=np.int8)
    session.env.move_size = int(np.count_nonzero(session.env.board))
    session.env.last_action = preset.get("last_action", -1)
    session.env.done = False
    session.env.winner = None
    session.env.current_player = preset.get(
        "current_player"
    ) or infer_current_player_from_board(board)
    if session.env.last_action is None:
        session.env.last_action = -1


def cancel_ai_task(session: GameSession) -> None:
    if session.ai_task and not session.ai_task.done():
        session.ai_task.cancel()
    session.ai_task = None
    session.ai_paused = False


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "board_size": BOARD_SIZE,
        "mcts_iterations": MCTS_ITERATIONS,
    }


@app.post("/api/validation/save")
async def save_validation_sample(sample: ValidationSample) -> dict[str, Any]:
    try:
        validate_state_payload(sample.state, sample.board_size)
        best_actions = normalize_best_action(sample.best_action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    max_action = sample.board_size * sample.board_size
    if any(action < 0 or action >= max_action for action in best_actions):
        raise HTTPException(
            status_code=400,
            detail=f"best_action must be in range [0, {max_action - 1}]",
        )

    normalized_state = normalize_state_payload(sample.state, sample.board_size)
    entry_id = uuid.uuid4().hex
    entry = {
        "id": entry_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "board_size": sample.board_size,
        "state": normalized_state,
        "best_action": best_actions,
        "meta": sample.meta or {},
    }

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    with VALIDATION_FILE.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    return {"status": "ok", "id": entry_id}


@app.get("/api/validation/list")
async def list_validation_samples(limit: int = 50) -> dict[str, Any]:
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be >= 1")
    items = []
    for entry in read_validation_entries():
        items.append(
            {
                "id": entry.get("id"),
                "created_at": entry.get("created_at"),
                "board_size": entry.get("board_size"),
                "best_action": entry.get("best_action", []),
                "meta": entry.get("meta", {}),
            }
        )
    items = list(reversed(items))[:limit]
    return {"items": items}


@app.get("/api/validation/get")
async def get_validation_sample(id: str) -> dict[str, Any]:
    for entry in read_validation_entries():
        if entry.get("id") == id:
            return entry
    raise HTTPException(status_code=404, detail="sample not found")


@app.post("/api/validation/update")
async def update_validation_sample(payload: ValidationUpdate) -> dict[str, Any]:
    entries = read_validation_entries()
    if not entries:
        raise HTTPException(status_code=404, detail="validation set not found")

    idx = next(
        (i for i, entry in enumerate(entries) if entry.get("id") == payload.id),
        None,
    )
    if idx is None:
        raise HTTPException(status_code=404, detail="sample not found")

    entry = entries[idx]
    board_size = payload.board_size or entry.get("board_size", BOARD_SIZE)

    if payload.board_size is not None and payload.state is None:
        raise HTTPException(
            status_code=400, detail="board_size update requires state payload"
        )

    if payload.state is not None:
        try:
            validate_state_payload(payload.state, board_size)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        entry["state"] = normalize_state_payload(payload.state, board_size)
        entry["board_size"] = board_size

    if payload.best_action is not None:
        try:
            best_actions = normalize_best_action(payload.best_action)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        max_action = board_size * board_size
        if any(action < 0 or action >= max_action for action in best_actions):
            raise HTTPException(
                status_code=400,
                detail=f"best_action must be in range [0, {max_action - 1}]",
            )
        entry["best_action"] = best_actions

    if payload.meta is not None:
        entry["meta"] = payload.meta

    entries[idx] = entry
    write_validation_entries(entries)
    return {"status": "ok", "id": entry.get("id")}


@app.post("/api/validation/delete")
async def delete_validation_sample(payload: ValidationDelete) -> dict[str, Any]:
    entries = read_validation_entries()
    if not entries:
        raise HTTPException(status_code=404, detail="validation set not found")

    next_entries = [entry for entry in entries if entry.get("id") != payload.id]
    if len(next_entries) == len(entries):
        raise HTTPException(status_code=404, detail="sample not found")

    write_validation_entries(next_entries)
    return {"status": "ok", "id": payload.id}


@app.websocket("/ws/play")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "new_game":
                existing = manager.sessions.get(websocket)
                if existing is not None:
                    cancel_ai_task(existing)
                try:
                    human_player = parse_human_player(data)
                    preset = parse_preset(data)
                except ValueError as exc:
                    await manager.send_json(
                        {"type": "error", "message": str(exc)}, websocket
                    )
                    continue
                session = manager.new_session(
                    websocket, human_player, mode="human_vs_ai", preset=preset
                )
                await manager.send_json(game_state_payload(session), websocket)
                if (
                    session.env.current_player != session.human_player
                    and not session.env.done
                ):
                    await manager.send_json(
                        {"type": "status", "message": "ai_thinking"}, websocket
                    )
                    await apply_ai_move(session)
                    await manager.send_json(game_state_payload(session), websocket)
                continue

            if action == "ai_vs_ai":
                existing = manager.sessions.get(websocket)
                if existing is not None:
                    cancel_ai_task(existing)
                try:
                    preset = parse_preset(data)
                except ValueError as exc:
                    await manager.send_json(
                        {"type": "error", "message": str(exc)}, websocket
                    )
                    continue
                delay_ms = int(data.get("delay_ms", 0))
                pause_on_start = bool(data.get("pause_on_start", False))
                session = manager.new_session(
                    websocket, human_player=0, mode="ai_vs_ai", preset=preset
                )
                session.ai_stop = False
                session.ai_paused = pause_on_start
                await manager.send_json(game_state_payload(session), websocket)
                session.ai_task = asyncio.create_task(
                    run_ai_vs_ai(session, websocket, delay_ms=delay_ms)
                )
                if pause_on_start:
                    await manager.send_json(
                        {"type": "status", "message": "ai_paused"}, websocket
                    )
                continue

            if action == "ai_pause":
                session = manager.sessions.get(websocket)
                if session is None or session.mode != "ai_vs_ai":
                    await manager.send_json(
                        {"type": "error", "message": "No active AI vs AI session."},
                        websocket,
                    )
                    continue
                session.ai_paused = True
                await manager.send_json(
                    {"type": "status", "message": "ai_paused"}, websocket
                )
                continue

            if action == "ai_resume":
                session = manager.sessions.get(websocket)
                if session is None or session.mode != "ai_vs_ai":
                    await manager.send_json(
                        {"type": "error", "message": "No active AI vs AI session."},
                        websocket,
                    )
                    continue
                session.ai_paused = False
                await manager.send_json(
                    {"type": "status", "message": "ai_resumed"}, websocket
                )
                continue

            if action == "ai_stop":
                session = manager.sessions.get(websocket)
                if session is None or session.mode != "ai_vs_ai":
                    await manager.send_json(
                        {"type": "error", "message": "No active AI vs AI session."},
                        websocket,
                    )
                    continue
                session.ai_stop = True
                cancel_ai_task(session)
                session.ai_stop = True
                await manager.send_json(
                    {"type": "status", "message": "ai_stopped"}, websocket
                )
                continue

            if action == "ai_step":
                session = manager.sessions.get(websocket)
                if session is None or session.mode != "ai_vs_ai":
                    await manager.send_json(
                        {"type": "error", "message": "No active AI vs AI session."},
                        websocket,
                    )
                    continue
                if (
                    session.ai_task
                    and not session.ai_task.done()
                    and not session.ai_paused
                ):
                    await manager.send_json(
                        {"type": "error", "message": "Pause AI before stepping."},
                        websocket,
                    )
                    continue
                if session.env.done:
                    await manager.send_json(
                        {"type": "error", "message": "Game is finished."},
                        websocket,
                    )
                    continue
                await apply_ai_move(session)
                await manager.send_json(game_state_payload(session), websocket)
                continue

            session = manager.sessions.get(websocket)
            if session is None:
                await manager.send_json(
                    {
                        "type": "error",
                        "message": "No active session. Send action=new_game first.",
                    },
                    websocket,
                )
                continue

            if action != "move":
                await manager.send_json(
                    {"type": "error", "message": "Unknown action."}, websocket
                )
                continue

            if session.env.done:
                await manager.send_json(
                    {"type": "error", "message": "Game is finished."}, websocket
                )
                continue

            if session.mode == "ai_vs_ai":
                await manager.send_json(
                    {"type": "error", "message": "AI vs AI mode is read-only."},
                    websocket,
                )
                continue

            if session.env.current_player != session.human_player:
                await manager.send_json(
                    {"type": "error", "message": "Not your turn."}, websocket
                )
                continue

            x = data.get("x")
            y = data.get("y")
            if x is None or y is None:
                await manager.send_json(
                    {"type": "error", "message": "Missing move coordinates."}, websocket
                )
                continue

            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                await manager.send_json(
                    {"type": "error", "message": "Move out of bounds."}, websocket
                )
                continue

            if not session.env._is_valid_move(x, y):
                await manager.send_json(
                    {"type": "error", "message": "Invalid move."}, websocket
                )
                continue

            move_player = session.env.current_player
            action_index = x * BOARD_SIZE + y
            session.env.step(action_index)
            session.last_move = {"x": x, "y": y, "player": move_player}
            await manager.send_json(game_state_payload(session), websocket)

            if session.env.done:
                continue

            await manager.send_json(
                {"type": "status", "message": "ai_thinking"}, websocket
            )
            await apply_ai_move(session)
            await manager.send_json(game_state_payload(session), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from gomoku.gomoku_env import GomokuEnvSimple
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcts import ZeroMCTS

BOARD_SIZE = 7
DEFAULT_MODEL_PATH = os.getenv(
    "GOMOKU_MODEL_PATH",
    # "gomoku/continue_model/policy_step_940000.pth",
    # "models/gomoku_zero_9_pre5/policy_step_990000.pth",
    "models/gomoku_zero_9_lab_2/policy_step_80000.pth",
)
MCTS_ITERATIONS = int(os.getenv("GOMOKU_MCTS_ITERS", "400"))
MCTS_PUCT = float(os.getenv("GOMOKU_MCTS_PUCT", "2.0"))
MAX_WORKERS = int(os.getenv("GOMOKU_AI_WORKERS", "2"))


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
        self.sessions.pop(websocket, None)

    async def send_json(self, message: dict[str, Any], websocket: WebSocket):
        await websocket.send_json(message)

    def new_session(self, websocket: WebSocket, human_player: int) -> GameSession:
        env = GomokuEnvSimple(board_size=BOARD_SIZE)
        env.reset()
        session = GameSession(env=env, human_player=human_player)
        self.sessions[websocket] = session
        return session


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


def parse_human_player(payload: dict[str, Any]) -> int:
    human_player = int(payload.get("human_player", 2))
    if human_player not in (1, 2):
        raise ValueError("human_player must be 1 or 2")
    return human_player


@app.get("/health")
async def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "board_size": BOARD_SIZE,
        "mcts_iterations": MCTS_ITERATIONS,
    }


@app.websocket("/ws/play")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "new_game":
                try:
                    human_player = parse_human_player(data)
                except ValueError as exc:
                    await manager.send_json({"type": "error", "message": str(exc)}, websocket)
                    continue
                session = manager.new_session(websocket, human_player)
                await manager.send_json(game_state_payload(session), websocket)
                if session.env.current_player != session.human_player and not session.env.done:
                    await manager.send_json({"type": "status", "message": "ai_thinking"}, websocket)
                    await apply_ai_move(session)
                    await manager.send_json(game_state_payload(session), websocket)
                continue

            session = manager.sessions.get(websocket)
            if session is None:
                await manager.send_json(
                    {"type": "error", "message": "No active session. Send action=new_game first."},
                    websocket,
                )
                continue

            if action != "move":
                await manager.send_json({"type": "error", "message": "Unknown action."}, websocket)
                continue

            if session.env.done:
                await manager.send_json({"type": "error", "message": "Game is finished."}, websocket)
                continue

            if session.env.current_player != session.human_player:
                await manager.send_json({"type": "error", "message": "Not your turn."}, websocket)
                continue

            x = data.get("x")
            y = data.get("y")
            if x is None or y is None:
                await manager.send_json({"type": "error", "message": "Missing move coordinates."}, websocket)
                continue

            if not (0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE):
                await manager.send_json({"type": "error", "message": "Move out of bounds."}, websocket)
                continue

            if not session.env._is_valid_move(x, y):
                await manager.send_json({"type": "error", "message": "Invalid move."}, websocket)
                continue

            move_player = session.env.current_player
            action_index = x * BOARD_SIZE + y
            session.env.step(action_index)
            session.last_move = {"x": x, "y": y, "player": move_player}
            await manager.send_json(game_state_payload(session), websocket)

            if session.env.done:
                continue

            await manager.send_json({"type": "status", "message": "ai_thinking"}, websocket)
            await apply_ai_move(session)
            await manager.send_json(game_state_payload(session), websocket)

    except WebSocketDisconnect:
        manager.disconnect(websocket)

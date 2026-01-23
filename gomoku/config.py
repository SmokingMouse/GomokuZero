from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class AppConfig:
    board_size: int = 9
    model_path: str = "models/gomoku_zero_9_lab_5/policy_step_990000.pth"
    mcts_iterations: int = 1200
    mcts_puct: float = 2.0
    max_workers: int = 2
    cors_origins: list[str] = field(default_factory=lambda: ["*"])


@dataclass(frozen=True)
class GuiConfig:
    board_size: int = 15
    square_size: int = 40
    margin: int = 40
    model_path: str = "models/gomoku_zero_9_lab_4/policy_step_100000.pth"


@dataclass(frozen=True)
class TrainerConfig:
    board_size: int = 9
    lr: float = 5e-4
    save_per_steps: int = 10000
    cpus: int = 16
    device: str = "cuda"
    seed: int = 42
    mcts_class: str = "zero_mcts_rs.ZeroMCGS"
    temperature_moves: int = 30
    self_play_device: str = "cpu"
    use_batch_inference: bool = False
    use_shared_policy_server: bool = True
    policy_server_device: str | None = None
    policy_server_concurrency: int = 64
    mcts_eval_batch_size: int = 4
    mcts_virtual_loss: float = 1.0
    mcts_max_children: int = 0
    batch_infer_size: int = 128
    batch_infer_wait_ms: float = 4.0
    batch_infer_queue: int = 4096
    batch_infer_enqueue_ms: float = 1000.0
    batch_infer_stats_sec: float = 1.0
    profiles: dict[int, "TrainerProfile"] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainerProfile:
    steps: int = 1000000
    buffer_size: int = 60000
    self_play_per_steps: int = 250
    self_play_num: int = 32
    eval_steps: int = 100000
    num_workers: int = 16
    games_per_worker: int = 2
    alpha: int = 1
    itermax: int = 200
    validation_eval_step: int = 5000
    validation_top_k: int = 5
    validation_path: str = "gomoku/validation_sets/validation_set.jsonl"
    lab_name: str = "gomoku_zero_9_lab_5"
    comment: str = "mcgs"
    batch_size: int = 256
    threshold: float = 0.2


@dataclass(frozen=True)
class GomokuConfig:
    app: AppConfig
    gui: GuiConfig
    trainer: TrainerConfig


def _resolve_config_path(path: Path | None) -> Path:
    if path is not None:
        return path
    return Path(__file__).resolve().parent / "config.toml"


def _coerce_cors_origins(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return ["*"]


def load_config(path: Path | None = None) -> GomokuConfig:
    config_path = _resolve_config_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    app_data = data.get("app", {})
    gui_data = data.get("gui", {})
    trainer_data = data.get("trainer", {})

    app_defaults = AppConfig()
    app = AppConfig(
        board_size=int(app_data.get("board_size", app_defaults.board_size)),
        model_path=str(app_data.get("model_path", app_defaults.model_path)),
        mcts_iterations=int(
            app_data.get("mcts_iterations", app_defaults.mcts_iterations)
        ),
        mcts_puct=float(app_data.get("mcts_puct", app_defaults.mcts_puct)),
        max_workers=int(app_data.get("max_workers", app_defaults.max_workers)),
        cors_origins=_coerce_cors_origins(
            app_data.get("cors_origins", app_defaults.cors_origins)
        ),
    )

    gui = GuiConfig(
        board_size=int(gui_data.get("board_size", GuiConfig.board_size)),
        square_size=int(gui_data.get("square_size", GuiConfig.square_size)),
        margin=int(gui_data.get("margin", GuiConfig.margin)),
        model_path=str(gui_data.get("model_path", GuiConfig.model_path)),
    )

    trainer_defaults = TrainerConfig()
    profile_defaults = TrainerProfile()
    profiles_data = trainer_data.get("profiles", {})
    profiles: dict[int, TrainerProfile] = {}
    for key, value in profiles_data.items():
        profile_data = value or {}
        try:
            board_key = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trainer profile key: {key}") from exc
        profiles[board_key] = TrainerProfile(
            steps=int(profile_data.get("steps", profile_defaults.steps)),
            buffer_size=int(profile_data.get("buffer_size", profile_defaults.buffer_size)),
            self_play_per_steps=int(
                profile_data.get(
                    "self_play_per_steps", profile_defaults.self_play_per_steps
                )
            ),
            self_play_num=int(
                profile_data.get("self_play_num", profile_defaults.self_play_num)
            ),
            eval_steps=int(profile_data.get("eval_steps", profile_defaults.eval_steps)),
            num_workers=int(
                profile_data.get("num_workers", profile_defaults.num_workers)
            ),
            games_per_worker=int(
                profile_data.get("games_per_worker", profile_defaults.games_per_worker)
            ),
            alpha=int(profile_data.get("alpha", profile_defaults.alpha)),
            itermax=int(profile_data.get("itermax", profile_defaults.itermax)),
            validation_eval_step=int(
                profile_data.get(
                    "validation_eval_step", profile_defaults.validation_eval_step
                )
            ),
            validation_top_k=int(
                profile_data.get(
                    "validation_top_k", profile_defaults.validation_top_k
                )
            ),
            validation_path=str(
                profile_data.get("validation_path", profile_defaults.validation_path)
            ),
            lab_name=str(profile_data.get("lab_name", profile_defaults.lab_name)),
            comment=str(profile_data.get("comment", profile_defaults.comment)),
            batch_size=int(profile_data.get("batch_size", profile_defaults.batch_size)),
            threshold=float(
                profile_data.get("threshold", profile_defaults.threshold)
            ),
        )

    trainer = TrainerConfig(
        board_size=int(trainer_data.get("board_size", trainer_defaults.board_size)),
        lr=float(trainer_data.get("lr", trainer_defaults.lr)),
        save_per_steps=int(
            trainer_data.get("save_per_steps", trainer_defaults.save_per_steps)
        ),
        cpus=int(trainer_data.get("cpus", trainer_defaults.cpus)),
        device=str(trainer_data.get("device", trainer_defaults.device)),
        seed=int(trainer_data.get("seed", trainer_defaults.seed)),
        mcts_class=str(trainer_data.get("mcts_class", trainer_defaults.mcts_class)),
        temperature_moves=int(
            trainer_data.get("temperature_moves", trainer_defaults.temperature_moves)
        ),
        self_play_device=str(
            trainer_data.get("self_play_device", trainer_defaults.self_play_device)
        ),
        use_batch_inference=bool(
            trainer_data.get(
                "use_batch_inference", trainer_defaults.use_batch_inference
            )
        ),
        use_shared_policy_server=bool(
            trainer_data.get(
                "use_shared_policy_server",
                trainer_defaults.use_shared_policy_server,
            )
        ),
        policy_server_device=trainer_data.get(
            "policy_server_device", trainer_defaults.policy_server_device
        ),
        policy_server_concurrency=int(
            trainer_data.get(
                "policy_server_concurrency",
                trainer_defaults.policy_server_concurrency,
            )
        ),
        mcts_eval_batch_size=int(
            trainer_data.get(
                "mcts_eval_batch_size", trainer_defaults.mcts_eval_batch_size
            )
        ),
        mcts_virtual_loss=float(
            trainer_data.get("mcts_virtual_loss", trainer_defaults.mcts_virtual_loss)
        ),
        mcts_max_children=int(
            trainer_data.get("mcts_max_children", trainer_defaults.mcts_max_children)
        ),
        batch_infer_size=int(
            trainer_data.get("batch_infer_size", trainer_defaults.batch_infer_size)
        ),
        batch_infer_wait_ms=float(
            trainer_data.get(
                "batch_infer_wait_ms", trainer_defaults.batch_infer_wait_ms
            )
        ),
        batch_infer_queue=int(
            trainer_data.get(
                "batch_infer_queue", trainer_defaults.batch_infer_queue
            )
        ),
        batch_infer_enqueue_ms=float(
            trainer_data.get(
                "batch_infer_enqueue_ms", trainer_defaults.batch_infer_enqueue_ms
            )
        ),
        batch_infer_stats_sec=float(
            trainer_data.get(
                "batch_infer_stats_sec", trainer_defaults.batch_infer_stats_sec
            )
        ),
        profiles=profiles,
    )

    if not trainer.policy_server_device:
        trainer = TrainerConfig(
            **{
                **trainer.__dict__,
                "policy_server_device": trainer.self_play_device,
            }
        )

    return GomokuConfig(app=app, gui=gui, trainer=trainer)

# %%
from dataclasses import dataclass

from gomoku.player import (
    MCTSConfig,
    PolicyRuntime,
    SelfPlaySettings,
    run_self_play,
    is_rs_mcts_class,
)
from gomoku.policy import ZeroPolicy
from gomoku.batched_inference import BatchPolicyRunner
from gomoku.policy_server import PolicyServer
import ray
import numpy as np
from gomoku.utils import timer
import torch


# 1. 初始化 Ray
@dataclass(frozen=True)
class InferenceConfig:
    use_batch_inference: bool = True
    use_shared_policy_server: bool = True
    policy_server_device: str | None = None
    policy_server_concurrency: int = 64
    batch_size: int = 64
    max_wait_ms: float = 2.0
    max_queue_size: int = 4096
    enqueue_timeout_ms: float = 1000.0
    stats_interval_sec: float = 1.0


@ray.remote
class SelfPlayWorker:
    def __init__(
        self,
        board_size,
        device,
        use_batch_inference: bool = True,
        use_shared_policy_server: bool = True,
        policy_server=None,
        mcts_eval_batch_size: int = 1,
        mcts_virtual_loss: float = 1.0,
        mcts_max_children: int = 0,
        batch_size: int = 64,
        max_wait_ms: float = 2.0,
        max_queue_size: int = 4096,
        enqueue_timeout_ms: float = 1000.0,
        stats_interval_sec: float = 1.0,
        mcts_config: MCTSConfig | None = None,
        inference_config: InferenceConfig | None = None,
    ):
        self.policy = ZeroPolicy(board_size=board_size).to(device)
        self.device = device
        self.board_size = board_size
        self.policy_server = policy_server
        self.mcts_config = mcts_config or MCTSConfig(
            eval_batch_size=mcts_eval_batch_size,
            virtual_loss=mcts_virtual_loss,
            max_children=mcts_max_children,
        )
        self.inference_config = inference_config or InferenceConfig(
            use_batch_inference=use_batch_inference,
            use_shared_policy_server=use_shared_policy_server,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            max_queue_size=max_queue_size,
            enqueue_timeout_ms=enqueue_timeout_ms,
            stats_interval_sec=stats_interval_sec,
        )
        self.policy_runner = None
        use_rs_mcts = is_rs_mcts_class(self.mcts_config.mcts_class)
        if (
            self.inference_config.use_batch_inference
            and not use_rs_mcts
            and self.inference_config.use_shared_policy_server
            and self.policy_server is not None
        ):
            self.policy_runner = RemotePolicyRunner(self.policy_server, self.device)
        elif self.inference_config.use_batch_inference and not use_rs_mcts:
            self.policy_runner = BatchPolicyRunner(
                self.policy,
                device=self.device,
                batch_size=self.inference_config.batch_size,
                max_wait_ms=self.inference_config.max_wait_ms,
                max_queue_size=self.inference_config.max_queue_size,
                enqueue_timeout_ms=self.inference_config.enqueue_timeout_ms,
                stats_interval_sec=self.inference_config.stats_interval_sec,
            )
        print(f"Worker actor initialized on {self.device}")

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)
        self.policy.eval()

    def play_game(self, itermax, temperature_moves: int = 30):
        settings = SelfPlaySettings(
            board_size=self.board_size,
            itermax=itermax,
            temperature_moves=temperature_moves,
        )
        runtime = PolicyRuntime(
            policy_runner=self.policy_runner,
            policy_server=self.policy_server,
        )
        r = run_self_play(
            self.policy,
            self.device,
            settings=settings,
            mcts_config=self.mcts_config,
            runtime=runtime,
        )
        return r

    def close(self):
        if self.policy_runner is not None:
            self.policy_runner.close()
            self.policy_runner = None


class RemotePolicyRunner:
    def __init__(self, policy_server, device: str):
        self.policy_server = policy_server
        self.device = device

    def predict(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        logits_np, value_np = ray.get(self.policy_server.predict.remote(obs))
        logits = torch.from_numpy(logits_np).to(self.device)
        value = torch.from_numpy(value_np).to(self.device)
        return logits, value

    def close(self):
        return None


@timer
def gather_selfplay_games(
    policy,
    device,
    board_size=9,
    itermax=200,
    temperature_moves: int = 30,
    num_workers=6,
    games_per_worker=5,
    use_batch_inference: bool = True,
    use_shared_policy_server: bool = True,
    policy_server_device: str | None = None,
    policy_server_concurrency: int = 64,
    mcts_eval_batch_size: int = 1,
    mcts_virtual_loss: float = 1.0,
    mcts_max_children: int = 0,
    batch_size: int = 64,
    max_wait_ms: float = 2.0,
    max_queue_size: int = 4096,
    enqueue_timeout_ms: float = 1000.0,
    stats_interval_sec: float = 1.0,
    mcts_class: type | None = None,
):
    """
    在单个 worker 上进行 self-play，返回游戏数据。
    """
    policy.eval()

    weights_ref = ray.put({k: v.cpu() for k, v in policy.state_dict().items()})
    # ray.put(policy)

    mcts_config = MCTSConfig(
        mcts_class=mcts_class,
        eval_batch_size=mcts_eval_batch_size,
        virtual_loss=mcts_virtual_loss,
        max_children=mcts_max_children,
    )
    inference_config = InferenceConfig(
        use_batch_inference=use_batch_inference,
        use_shared_policy_server=use_shared_policy_server,
        policy_server_device=policy_server_device,
        policy_server_concurrency=policy_server_concurrency,
        batch_size=batch_size,
        max_wait_ms=max_wait_ms,
        max_queue_size=max_queue_size,
        enqueue_timeout_ms=enqueue_timeout_ms,
        stats_interval_sec=stats_interval_sec,
    )

    policy_server = None
    if (
        inference_config.use_batch_inference
        and inference_config.use_shared_policy_server
    ):
        server_device = inference_config.policy_server_device or device
        policy_server_options = {
            "max_concurrency": inference_config.policy_server_concurrency,
        }
        if str(server_device).startswith("cuda"):
            policy_server_options["num_gpus"] = 1

        policy_server = PolicyServer.options(**policy_server_options).remote(
            board_size=board_size,
            device=server_device,
            batch_size=inference_config.batch_size,
            max_wait_ms=inference_config.max_wait_ms,
            max_queue_size=inference_config.max_queue_size,
            enqueue_timeout_ms=inference_config.enqueue_timeout_ms,
            stats_interval_sec=inference_config.stats_interval_sec,
        )
        ray.get(policy_server.set_weights.remote(weights_ref))

    workers = [
        SelfPlayWorker.remote(
            board_size,
            device,
            policy_server=policy_server,
            mcts_config=mcts_config,
            inference_config=inference_config,
        )
        for _ in range(num_workers)
    ]
    try:
        set_weights_tasks = [
            worker.set_weights.remote(weights_ref) for worker in workers
        ]
        ray.get(set_weights_tasks)  # 等待权重设置完成
        game_futures = [
            worker.play_game.remote(itermax, temperature_moves)
            for _ in range(games_per_worker)
            for worker in workers
        ]  # 简化版任务分配

        games_results = ray.get(game_futures)
        games = []
        for result in games_results:
            if isinstance(result, list):
                games.extend(result)
            else:
                games.append(result)
        return games
    finally:
        close_tasks = [worker.close.remote() for worker in workers]
        ray.get(close_tasks)
        if policy_server is not None:
            ray.get(policy_server.close.remote())
            ray.kill(policy_server)
        for worker in workers:
            ray.kill(worker)

    # games = ray.get(game_futures)
    # return games


def get_symmetric_data(state, pi, board_size=9):
    """
    生成棋盘状态和策略的对称增强样本（旋转+翻转）
    假设state形状为(C, 9, 9)，pi长度为81（对应9x9棋盘）
    """
    # 输入校验
    assert isinstance(state, np.ndarray), "state应为numpy数组"
    assert state.ndim == 3 and state.shape[1:] == (board_size, board_size), (
        "state形状应为(C, 9, 9)"
    )
    assert len(pi) == board_size * board_size, f"pi长度必须为81，实际为{len(pi)}"

    pi_board = np.asarray(pi).reshape((board_size, board_size))
    augmented_samples = []

    # 4种旋转（0/90/180/270度）+ 水平翻转，共8种对称
    for i in range(4):
        # 旋转
        rotated_state = np.rot90(state, axes=(1, 2), k=i)  # 仅旋转空间维度
        rotated_pi = np.rot90(pi_board, k=i)
        augmented_samples.append((rotated_state, rotated_pi.flatten().tolist()))

        # 旋转后翻转
        flipped_state = np.flip(rotated_state, axis=2)
        flipped_pi = np.flip(rotated_pi, axis=1)
        augmented_samples.append((flipped_state, flipped_pi.flatten().tolist()))

    return augmented_samples


# 2. 主进程逻辑
if __name__ == "__main__":
    total_games = 9
    iter_max = 40
    board_size = 9
    device = "cpu"
    num_workers = 6

    ray.init(num_cpus=6)
    policy = ZeroPolicy(board_size=board_size).to(device)
    games = gather_selfplay_games(policy, device, itermax=iter_max)
    ray.shutdown()

    for game in games:
        print(f"Game states: {game['rewards'][0]}")

    print(f"Generated {len(games)} games.")

# %%

import numpy as np
import ray
import torch

from gomoku.batched_inference import BatchPolicyRunner
from gomoku.policy import ZeroPolicy


@ray.remote
class PolicyServer:
    def __init__(
        self,
        board_size: int,
        device: str,
        batch_size: int = 64,
        max_wait_ms: float = 2.0,
        max_queue_size: int = 4096,
        enqueue_timeout_ms: float = 1000.0,
        stats_interval_sec: float = 1.0,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for PolicyServer but not available.")
        self.policy = ZeroPolicy(board_size=board_size).to(device)
        self.policy.eval()
        self.runner = BatchPolicyRunner(
            self.policy,
            device=device,
            batch_size=batch_size,
            max_wait_ms=max_wait_ms,
            max_queue_size=max_queue_size,
            enqueue_timeout_ms=enqueue_timeout_ms,
            stats_interval_sec=stats_interval_sec,
        )

    def set_weights(self, weights) -> None:
        self.policy.load_state_dict(weights)
        self.policy.eval()

    def predict(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        policy_logits, value = self.runner.predict(obs)
        return (
            policy_logits.detach().cpu().numpy(),
            value.detach().cpu().numpy(),
        )

    def close(self) -> None:
        self.runner.close()

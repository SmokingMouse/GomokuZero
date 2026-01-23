import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class _BatchRequest:
    obs: np.ndarray
    event: threading.Event
    result: Optional[tuple[torch.Tensor, torch.Tensor]] = None
    error: Optional[BaseException] = None
    start_time: float = 0.0


class BatchPolicyRunner:
    """Batch policy inference across multiple callers."""

    def __init__(
        self,
        policy,
        device: Optional[str] = None,
        batch_size: int = 64,
        max_wait_ms: float = 2.0,
        max_queue_size: int = 4096,
        enqueue_timeout_ms: float = 1000.0,
        stats_interval_sec: float = 1.0,
    ) -> None:
        self.policy = policy
        self.policy.eval()
        self.device = device or next(policy.parameters()).device
        self.batch_size = max(1, batch_size)
        self.max_wait_ms = max(0.0, max_wait_ms)
        self.max_queue_size = max(0, max_queue_size)
        self.enqueue_timeout_ms = max(0.0, enqueue_timeout_ms)
        self.stats_interval_sec = max(0.0, stats_interval_sec)

        if self.max_queue_size > 0:
            self._queue = queue.Queue(maxsize=self.max_queue_size)
        else:
            self._queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._stats_lock = threading.Lock()
        self._last_log_time = time.monotonic()
        self._total_requests = 0
        self._total_batches = 0
        self._total_batch_size = 0
        self._total_infer_time = 0.0
        self._total_latency = 0.0
        self._queue_full = 0
        self._worker.start()

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put(_BatchRequest(np.empty(0), threading.Event()), timeout=0.1)
        except queue.Full:
            pass
        self._worker.join(timeout=1.0)

    def predict(self, obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        event = threading.Event()
        start_time = time.monotonic()
        request = _BatchRequest(obs=obs, event=event, start_time=start_time)
        try:
            timeout = None
            if self.enqueue_timeout_ms > 0:
                timeout = self.enqueue_timeout_ms / 1000.0
            self._queue.put(request, timeout=timeout)
        except queue.Full as exc:
            with self._stats_lock:
                self._queue_full += 1
            raise TimeoutError("BatchPolicyRunner queue is full.") from exc
        event.wait()
        latency = time.monotonic() - start_time
        with self._stats_lock:
            self._total_requests += 1
            self._total_latency += latency
        if request.error is not None:
            raise request.error
        if request.result is None:
            raise RuntimeError("BatchPolicyRunner returned no result.")
        return request.result

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self._queue.get(timeout=0.1)
            except queue.Empty:
                # print("BatchPolicyRunner waiting for requests...")
                continue
            if self._stop_event.is_set():
                break

            batch = [first]
            start = time.monotonic()
            while len(batch) < self.batch_size:
                timeout = self.max_wait_ms / 1000.0 - (time.monotonic() - start)
                if timeout <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=timeout))
                except queue.Empty:
                    # print("BatchPolicyRunner batch timeout reached.")

                    break

            try:
                obs_batch = np.stack([req.obs for req in batch], axis=0)
                torch_x = torch.from_numpy(obs_batch).float().to(self.device)
                infer_start = time.monotonic()
                with torch.no_grad():
                    policy_logits, value = self.policy(torch_x)
                infer_time = time.monotonic() - infer_start
                for i, req in enumerate(batch):
                    req.result = (
                        policy_logits[i].unsqueeze(0),
                        value[i].unsqueeze(0),
                    )
                with self._stats_lock:
                    self._total_batches += 1
                    self._total_batch_size += len(batch)
                    self._total_infer_time += infer_time
            except BaseException as exc:
                # print("BatchPolicyRunner encountered an error: {}".format(exc))
                for req in batch:
                    req.error = exc
            finally:
                for req in batch:
                    req.event.set()
            # print("BatchPolicyRunner processed batch of size {}".format(len(batch)))

            self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        if self.stats_interval_sec <= 0:
            return
        now = time.monotonic()
        with self._stats_lock:
            if now - self._last_log_time < self.stats_interval_sec:
                return
            elapsed = now - self._last_log_time
            total_reqs = self._total_requests
            total_batches = self._total_batches
            total_batch_size = self._total_batch_size
            total_infer_time = self._total_infer_time
            total_latency = self._total_latency
            queue_full = self._queue_full
            self._total_requests = 0
            self._total_batches = 0
            self._total_batch_size = 0
            self._total_infer_time = 0.0
            self._total_latency = 0.0
            self._queue_full = 0
            self._last_log_time = now

        avg_batch = (total_batch_size / total_batches) if total_batches else 0.0
        reqs_per_sec = total_reqs / elapsed if elapsed > 0 else 0.0
        avg_infer_ms = (
            (total_infer_time / total_batches) * 1000.0 if total_batches else 0.0
        )
        avg_latency_ms = (total_latency / total_reqs) * 1000.0 if total_reqs else 0.0
        try:
            qsize = self._queue.qsize()
        except NotImplementedError:
            # print("Queue size not available.")
            qsize = -1

        print(
            "[BatchPolicyRunner] req/s={:.1f} avg_batch={:.1f} "
            "avg_infer_ms={:.2f} avg_latency_ms={:.2f} qsize={} queue_full={}".format(
                reqs_per_sec,
                avg_batch,
                avg_infer_ms,
                avg_latency_ms,
                qsize,
                queue_full,
            )
        )

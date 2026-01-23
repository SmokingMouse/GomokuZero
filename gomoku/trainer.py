import os
from collections import deque

from gomoku.config import load_config
from gomoku.evaluate import (
    evaluate_validation_samples,
    load_validation_samples,
    resolve_validation_path,
)
from gomoku.worker import gather_selfplay_games, get_symmetric_data

import random
from gomoku.policy import ZeroPolicy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn.functional as F
import rich
import tqdm
import numpy as np
import ray

CONFIG = load_config()
TRAINER_CONFIG = CONFIG.trainer

self_play_device = TRAINER_CONFIG.self_play_device
use_batch_inference = TRAINER_CONFIG.use_batch_inference
use_shared_policy_server = TRAINER_CONFIG.use_shared_policy_server
policy_server_device = TRAINER_CONFIG.policy_server_device
policy_server_concurrency = TRAINER_CONFIG.policy_server_concurrency
mcts_eval_batch_size = TRAINER_CONFIG.mcts_eval_batch_size
mcts_virtual_loss = TRAINER_CONFIG.mcts_virtual_loss
mcts_max_children = TRAINER_CONFIG.mcts_max_children
mcts_class = TRAINER_CONFIG.mcts_class
batch_infer_size = TRAINER_CONFIG.batch_infer_size
batch_infer_wait_ms = TRAINER_CONFIG.batch_infer_wait_ms
batch_infer_queue = TRAINER_CONFIG.batch_infer_queue
batch_infer_enqueue_ms = TRAINER_CONFIG.batch_infer_enqueue_ms
batch_infer_stats_sec = TRAINER_CONFIG.batch_infer_stats_sec

temperature_moves = TRAINER_CONFIG.temperature_moves

board_size = TRAINER_CONFIG.board_size
lr = TRAINER_CONFIG.lr
save_per_steps = TRAINER_CONFIG.save_per_steps
cpus = TRAINER_CONFIG.cpus
device = TRAINER_CONFIG.device
seed = TRAINER_CONFIG.seed

profile = TRAINER_CONFIG.profiles.get(board_size)
if profile is None:
    raise ValueError(f"No trainer profile configured for board_size={board_size}")

lab_name = profile.lab_name
comment = profile.comment
batch_size = profile.batch_size
threshold = profile.threshold
steps = profile.steps
buffer_size = profile.buffer_size
self_play_per_steps = profile.self_play_per_steps
self_play_num = profile.self_play_num
eval_steps = profile.eval_steps
num_workers = profile.num_workers
games_per_worker = profile.games_per_worker
alpha = profile.alpha
itermax = profile.itermax
validation_eval_step = profile.validation_eval_step
validation_top_k = profile.validation_top_k
validation_path = profile.validation_path


# 1. gomoku_zero_ray
# 使用 ray 加速采样 ✅
# 2. gomoku_zero_ray_dirichlet
# 使用 dirichlet 噪声，提升多样性 ✅
# 3. gomoku_zero_resnet
# 使用 resnet 块 ✅
# 4. arena
# 用于评测质量 ✅
# 5. gomoku_zero_scheduler  ✅
# 用于调整学习率(ReduceLROnPlateau)
# 6. gomoku_zero_effective_zero ✅
# 修复 zero 的问题，并使效率提升
# 7. 样本优化（对称、使用最优模型生成） ✅
# 生成对局时，使用最优模型


def train(policy: ZeroPolicy, optimizor, replay_buffer):
    writer = SummaryWriter(f"runs/{lab_name}", comment=comment)

    exclude_list = [
        ".git",  # 排除整个 .git 目录
        "__pycache__",  # 排除 Python 缓存
        "*.pyc",  # 排除编译后的 Python 文件
        "*.o",  # 排除 C++ 编译的中间文件
        ".idea",  # 排除 IDE 的配置文件
        ".vscode",  # 排除 VSCode 的配置文件
        "checkpoints/",  # 排除模型权重文件目录 (如果很大)
        "runs/",  # 排除 TensorBoard 日志目录
        ".venv/",  # 排除虚拟环境目录
    ]

    ray.init(
        num_cpus=cpus,
        runtime_env={
            "excludes": exclude_list,
            "working_dir": None,  # <--- 关键：告诉 Ray 不要自动同步当前目录
        },
    )
    # scheduler = ReduceLROnPlateau(optimizor, 'min', patience=100, factor=0.5, min_lr=1e-4)
    scheduler = CosineAnnealingLR(optimizor, T_max=steps, eta_min=5e-5)
    # scheduler = MultiStepLR(optimizor, milestones=[0.5 * steps, 0.75 * steps], gamma=0.2)

    best_policy = ZeroPolicy(board_size=board_size)
    best_policy.load_state_dict(policy.state_dict())

    update_count = 0

    for step in tqdm.tqdm(range(steps)):
        policy.train()
        if step % self_play_per_steps == 0:
            with torch.no_grad():
                # 使用一定比例的最优模型进行 self-play
                # generate_model = best_policy if random.random() > threshold else policy
                generate_model = policy

                generate_model.eval()
                games = gather_selfplay_games(
                    generate_model,
                    self_play_device,
                    board_size=board_size,
                    itermax=itermax,
                    games_per_worker=games_per_worker,
                    num_workers=num_workers,
                    temperature_moves=temperature_moves,
                    use_batch_inference=use_batch_inference,
                    use_shared_policy_server=use_shared_policy_server,
                    policy_server_device=policy_server_device,
                    policy_server_concurrency=policy_server_concurrency,
                    mcts_eval_batch_size=mcts_eval_batch_size,
                    mcts_virtual_loss=mcts_virtual_loss,
                    mcts_max_children=mcts_max_children,
                    batch_size=batch_infer_size,
                    max_wait_ms=batch_infer_wait_ms,
                    max_queue_size=batch_infer_queue,
                    enqueue_timeout_ms=batch_infer_enqueue_ms,
                    stats_interval_sec=batch_infer_stats_sec,
                    mcts_class=mcts_class,
                )
                for game in games:
                    for i in range(len(game["states"])):
                        augmented_samples = get_symmetric_data(
                            game["states"][i], game["probs"][i], board_size=board_size
                        )
                        for state, pi in augmented_samples:
                            replay_buffer.append((state, pi, game["rewards"][i]))
            rich.print(f"Self play {self_play_num} times")

        # if step != 0 and step % eval_steps == 0:
        #     policy.eval()
        #     best_policy.eval()

        #     policy_cpu_copy = ZeroPolicy(board_size=board_size).to("cpu")
        #     policy_state_dict = policy.state_dict()
        #     policy_cpu_copy.load_state_dict(policy_state_dict)

        #     best_policy_cpu_copy = ZeroPolicy(board_size=board_size).to("cpu")
        #     best_policy_state_dict = best_policy.state_dict()
        #     best_policy_cpu_copy.load_state_dict(best_policy_state_dict)

        #     r = arena_parallel(
        #         policy_cpu_copy,
        #         best_policy_cpu_copy,
        #         games=48,
        #         board_size=board_size,
        #         num_cpus=cpus,
        #         eager=False,
        #         itermax=itermax,
        #     )

        #     win_rate = r["player1_win_rate"]
        #     best_policy.load_state_dict(policy.state_dict())
        #     if win_rate >= 0.55:
        #         update_count += 1

        #     writer.add_scalar("Train/win-rate", win_rate, step)
        #     writer.add_scalar("Train/update-count", update_count, step)

        if step % validation_eval_step == 0:
            samples = load_validation_samples(resolve_validation_path(validation_path))
            if samples:
                metrics = evaluate_validation_samples(
                    policy,
                    samples,
                    device=device,
                    top_k=validation_top_k,
                    difficulty=None,
                )
                writer.add_scalar("Validation/top1", metrics["top1_accuracy"], step)
                writer.add_scalar("Validation/topk", metrics["topk_accuracy"], step)
                writer.add_scalar("Validation/total", metrics["total"], step)

                for level in range(1, 6):
                    level_metrics = evaluate_validation_samples(
                        policy,
                        samples,
                        device=device,
                        top_k=validation_top_k,
                        difficulty=level,
                    )
                    if level_metrics["total"] == 0:
                        continue
                    writer.add_scalar(
                        f"Validation/level_{level}/top1",
                        level_metrics["top1_accuracy"],
                        step,
                    )
                    writer.add_scalar(
                        f"Validation/level_{level}/topk",
                        level_metrics["topk_accuracy"],
                        step,
                    )
                    writer.add_scalar(
                        f"Validation/level_{level}/total",
                        level_metrics["total"],
                        step,
                    )

        policy.train()
        batch = random.sample(replay_buffer, batch_size)

        states, probs, rewards = zip(*batch)
        states_np = np.array(states)
        probs_np = np.array(probs)

        # 2. 把单一的大 NumPy 数组 -> PyTorch Tensor，并发送到 GPU
        states = torch.from_numpy(states_np).float().to(device)
        probs = torch.from_numpy(probs_np).float().to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        optimizor.zero_grad()

        logits, value = policy(states)

        mse = F.mse_loss(value.squeeze(), rewards, reduce="mean")
        log_probs = F.log_softmax(logits, dim=-1)
        cse = -torch.sum(
            probs * log_probs, dim=1
        ).mean()  # 先在每个样本上求和，再在batch上求平均

        loss = alpha * mse + cse  # / batch_size
        loss.backward()
        optimizor.step()
        scheduler.step()

        probs_from_log = torch.exp(log_probs)
        entropy = -torch.sum(probs_from_log * log_probs, dim=1).mean()

        writer.add_scalar("Train/loss", loss.item(), step)
        writer.add_scalar("Train/mse", mse.item(), step)
        writer.add_scalar("Train/cse", cse.item(), step)
        writer.add_scalar("Train/entropy", entropy.item(), step)
        writer.add_scalar("Train/lr", optimizor.param_groups[0]["lr"], step)
        rich.print(
            f"step: {step}, loss: {loss.item()}, mse: {mse.item()}, cse: {cse.item()}, entropy: {entropy.item()}"
        )
        if step != 0 and step % save_per_steps == 0:
            torch.save(policy.state_dict(), f"models/{lab_name}/policy_step_{step}.pth")
            rich.print(f"Saved model at step {step}")


if __name__ == "__main__":
    rich.print(f"Start training Gomoku Zero with {steps} steps")
    rich.print(f"Learning rate: {lr}, Save every {save_per_steps} steps")
    rich.print(f"Board size: {board_size}")

    if not torch.cuda.is_available():
        rich.print("[red]CUDA is not available. Training will be slow.[/red]")

    if os.path.exists(f"models/{lab_name}") is False:
        os.makedirs(f"models/{lab_name}")

    random.seed(seed)
    buffer = deque(maxlen=buffer_size)

    policy = ZeroPolicy(board_size=board_size, num_blocks=2, base_channels=32)
    # policy.load_state_dict(
    #     torch.load(
    #         # "/home/zhangpeng.pada/GomokuZero/models/gomoku_zero_9_lab_1/policy_step_990000.pth"
    #         "/home/smokingmouse/python/ai/GomokuZero/models/gomoku_zero_15_lab_1/policy_step_60000.pth"
    #     )
    # )
    policy.to(device)
    optimizor = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    train(policy, optimizor, buffer)

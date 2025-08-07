# %%
import os
from collections import deque
from gomoku.player import Player, ZeroMCTSPlayer, arena_parallel
from gomoku.worker import gather_selfplay_games, get_symmetric_data
import random
from gomoku.policy import ZeroPolicy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, MultiStepLR
import torch
import torch.nn.functional as F
import rich
import tqdm
import numpy as np
import ray

board_size = 15
lr = 5e-4
steps = 200000
save_per_steps = 500
lab_name = 'gomoku_zero_15'
batch_size = 256

# 采一次，至少用一次
# 采一次，每条样本至少训练 3 次  采 40N 条，经过 M / 40N * circle 次过期，每次抽取 256 条, 每条至少训练 3 次
# 所有 256 * M  / 40N * Circle / M 
buffer_size = 200000
device = 'cuda'
cpus = os.cpu_count() // 2
self_play_per_steps = 150
self_play_num = 32
eval_steps = 600
games_per_worker = self_play_num // cpus
num_workers = cpus
itermax=200
seed=42

threshold=0.2

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
    writer = SummaryWriter(f'runs/{lab_name}')
    ray.init(num_cpus=cpus)
    # scheduler = ReduceLROnPlateau(optimizor, 'min', patience=100, factor=0.5, min_lr=1e-4)
    # scheduler = CosineAnnealingLR(optimizor, T_max=steps//2, eta_min=5e-5)
    scheduler = MultiStepLR(optimizor, milestones=[100000, 150000], gamma=0.2)

    best_policy = ZeroPolicy(board_size=board_size)
    best_policy.load_state_dict(policy.state_dict())

    update_count = 0

    for step in tqdm.tqdm(range(steps)):
        policy.train()
        if step % self_play_per_steps == 0:
            with torch.no_grad():
                # 使用一定比例的最优模型进行 self-play
                generate_model = best_policy if random.random() > threshold else policy

                generate_model.eval()
                games = gather_selfplay_games(generate_model, 'cpu', board_size=board_size, itermax=itermax, games_per_worker=games_per_worker, num_workers=num_workers)
                for game in games:
                    for i in range(len(game['states'])):
                        augmented_samples = get_symmetric_data(game['states'][i], game['probs'][i], board_size=board_size)
                        for state, pi in augmented_samples:
                            replay_buffer.append((
                                state,
                                pi,
                                game['rewards'][i]
                            ))
            rich.print(f'Self play {self_play_num} times')
        
        if step % eval_steps == 0:
            policy.eval()
            best_policy.eval()

            policy_cpu_copy = ZeroPolicy(board_size=board_size).to('cpu')
            policy_state_dict = policy.state_dict()
            policy_cpu_copy.load_state_dict(policy_state_dict)
            
            best_policy_cpu_copy = ZeroPolicy(board_size=board_size).to('cpu')
            best_policy_state_dict = best_policy.state_dict()
            best_policy_cpu_copy.load_state_dict(best_policy_state_dict)
            
            r = arena_parallel(
                policy_cpu_copy,
                best_policy_cpu_copy,
                games=50,
                board_size=board_size,
                num_cpus=cpus
            )

            win_rate = r['player1_win_rate']
            if win_rate >= 0.55:
                best_policy.load_state_dict(policy.state_dict())
                update_count += 1
            
            writer.add_scalar('Train/win-rate', win_rate, step)
            writer.add_scalar('Train/update-count', update_count, step)

        
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
        
        mse = F.mse_loss(value.squeeze(), rewards, reduce='mean')  
        log_probs = F.log_softmax(logits, dim=-1)
        cse = -torch.sum(probs * log_probs, dim=1).mean() # 先在每个样本上求和，再在batch上求平均

        loss = (mse + cse) #/ batch_size
        loss.backward()
        optimizor.step()
        scheduler.step()

        probs_from_log = torch.exp(log_probs)
        entropy = -torch.sum(probs_from_log * log_probs, dim=1).mean()

        writer.add_scalar('Train/loss', loss.item(), step)
        writer.add_scalar('Train/mse', mse.item(), step)
        writer.add_scalar('Train/cse', cse.item(), step)
        writer.add_scalar('Train/entropy', entropy.item(), step)
        writer.add_scalar('Train/lr', optimizor.param_groups[0]['lr'], step)
        rich.print(f'step: {step}, loss: {loss.item()}, mse: {mse.item()}, cse: {cse.item()}, entropy: {entropy.item()}')
        if step != 0 and step % save_per_steps == 0:
            torch.save(policy.state_dict(), f'models/{lab_name}/policy_step_{step}.pth')
            rich.print(f'Saved model at step {step}')

if __name__ == "__main__":
    rich.print(f'Start training Gomoku Zero with {steps} steps')
    rich.print(f'Learning rate: {lr}, Save every {save_per_steps} steps')
    rich.print(f'Board size: {board_size}')

    if not torch.cuda.is_available():
        rich.print("[red]CUDA is not available. Training will be slow.[/red]")

    if os.path.exists(f'models/{lab_name}') is False:
        os.makedirs(f'models/{lab_name}')
    
    random.seed(seed)
    buffer = deque(maxlen=buffer_size)

    policy = ZeroPolicy(board_size=board_size).to(device)
    optimizor = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    train(policy, optimizor, buffer)

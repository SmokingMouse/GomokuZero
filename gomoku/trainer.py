# %%
import os
from collections import deque
from gomoku.worker import gather_selfplay_games
import random
from gomoku.policy import ZeroPolicy
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
import rich
import tqdm
import numpy as np
import ray

board_size = 9
lr = 2e-3
steps = 15000
save_per_steps = 500
lab_name = 'gomoku_zero_lrscheduler'
batch_size = 256

# 采一次，至少用一次
# 采一次，每条样本至少训练 3 次  采 40N 条，经过 M / 40N * circle 次过期，每次抽取 256 条, 每条至少训练 3 次
# 所有 256 * M  / 40N * Circle / M 
buffer_size =  12000
device = 'cuda'
cpus = os.cpu_count() - 4
self_play_per_steps = 10
self_play_num = 20 
eval_steps = 10
games_per_worker = self_play_num // cpus
num_workers = cpus
itermax=200
seed=42

# 1. gomoku_zero_ray 
    # 使用 ray 加速采样 ✅
# 2. gomoku_zero_ray_dirichlet
    # 使用 dirichlet 噪声，提升多样性 ✅
# 3. gomoku_zero_resnet 
    # 使用 resnet 块 ✅
# 4. arena 
    # 用于评测质量 ✅
    # 生成对局时，使用最优模型
# 5. gomoku_zero_scheduler
    # 用于调整学习率(ReduceLROnPlateau)

def train(policy, optimizor, replay_buffer):
    writer = SummaryWriter(f'runs/{lab_name}')
    ray.init(num_cpus=cpus)
    scheduler = ReduceLROnPlateau(optimizor, 'min', patience=100, factor=0.5, min_lr=1e-4)
    train_set, val_set = [], []

    for step in tqdm.tqdm(range(steps)):
        policy.train()
        if step % self_play_per_steps == 0:
            with torch.no_grad():
                policy.eval()

                games = gather_selfplay_games(policy, 'cpu', itermax=itermax, games_per_worker=games_per_worker, num_workers=num_workers)

                for game in games:
                    for i in range(len(game['states'])):
                        replay_buffer.append((
                            game['states'][i],
                            game['probs'][i],
                            game['rewards'][i]
                        ))
                train_set, val_set = train_test_split(replay_buffer, test_size=0.1)

            rich.print(f'Self play {self_play_num} times')
        
        policy.eval()
        with torch.no_grad():
            val_loss = 0
            states, probs, rewards = zip(*val_set)
            states_np = np.array(states)
            probs_np = np.array(probs)
            states = torch.from_numpy(states_np).float().to(device)
            probs = torch.from_numpy(probs_np).float().to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

            logits, value = policy(states)
            mse = F.mse_loss(value.squeeze(), rewards, reduce='mean')  
            log_probs = F.log_softmax(logits, dim=-1)
            cse = -torch.sum(probs * log_probs, dim=1).mean() # 先在每个样本上求和，再在batch上求平均
            val_loss = mse.item() + cse.item()
            val_loss /= len(val_set)
            scheduler.step(val_loss)
            writer.add_scalar('Loss/val_loss', val_loss, step)
        
        # if len(replay_buffer) < batch_size:
            # continue
        policy.train()
        batch = random.sample(train_set, batch_size)

        states, probs, rewards = zip(*batch)
        states_np = np.array(states)
        probs_np = np.array(probs)

        # 2. 把单一的大 NumPy 数组 -> PyTorch Tensor，并发送到 GPU
        states = torch.from_numpy(states_np).float().to(device)
        probs = torch.from_numpy(probs_np).float().to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)

        optimizor.zero_grad()

        logits, value = policy(states)
        # batch_size = len(records['states'])
        
        mse = F.mse_loss(value.squeeze(), rewards, reduce='mean')  
        log_probs = F.log_softmax(logits, dim=-1)
        cse = -torch.sum(probs * log_probs, dim=1).mean() # 先在每个样本上求和，再在batch上求平均

        loss = (mse + cse) #/ batch_size
        loss.backward()
        optimizor.step()

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
    # Create directory if not exists
    train(policy, optimizor, buffer)

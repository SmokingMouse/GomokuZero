# %%
import os
from collections import deque
from gomoku.worker import gather_selfplay_games
import random
from gomoku.policy import ZeroPolicy
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import rich
import tqdm
import numpy as np
import ray

board_size = 9
lr = 1e-3
steps = 15000
save_per_steps = 200
lab_name = 'gomoku_zero_ray_dirichlet_800'
batch_size = 256
buffer_size = 20000
device = 'cuda'
cpus = os.cpu_count() // 2
self_play_per_steps = 50
self_play_num = 60
games_per_worker = self_play_num // cpus
num_workers = cpus
itermax=800


def train(policy, optimizor, replay_buffer):
    writer = SummaryWriter(f'runs/{lab_name}')
    ray.init(num_cpus=cpus)

    for step in tqdm.tqdm(range(steps)):
        policy.train()

        if step % self_play_per_steps == 0:
            with torch.no_grad():
                policy.eval()

                games = gather_selfplay_games(policy, 'cpu', itermax=itermax, games_per_worker=games_per_worker, num_workers=cpus)

                for game in games:
                    for i in range(len(game['states'])):
                        replay_buffer.append((
                            game['states'][i],
                            game['probs'][i],
                            game['rewards'][i]
                        ))

            rich.print(f'Self play {self_play_num} times')
        
        # if len(replay_buffer) < batch_size:
            # continue
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
        # batch_size = len(records['states'])
        
        mse = F.mse_loss(value.squeeze(), rewards, reduce='mean')  
        # cse = -torch.sum(probs * F.log_softmax(logits, dim=-1)) / batch_size
        log_probs = F.log_softmax(logits, dim=-1)
        cse = -torch.sum(probs * log_probs, dim=1).mean() # 先在每个样本上求和，再在batch上求平均

        loss = (mse + cse) #/ batch_size
        loss.backward()
        optimizor.step()

        writer.add_scalar('loss', loss.item(), step)
        writer.add_scalar('mse', mse.item(), step)
        writer.add_scalar('cse', cse.item(), step)
        rich.print(f'step: {step}, loss: {loss}, mse: {mse}, cse: {cse}')
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
    
    buffer = deque(maxlen=buffer_size)

    policy = ZeroPolicy(board_size=board_size).to(device)
    optimizor = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    # Create directory if not exists
    train(policy, optimizor, buffer)

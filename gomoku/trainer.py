# %%
import os
from collections import deque
import random
from gomoku.gomoku_env import  GomokuEnvSimple
from gomoku.player import  ZeroMCTSPlayer
from gomoku.policy import ZeroPolicy
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
import rich
import tqdm
import numpy as np

# %% 
def self_play(policy):
    game = GomokuEnvSimple()
    player1 = ZeroMCTSPlayer(game, policy, itermax=800)
    player2 = ZeroMCTSPlayer(game, policy, itermax=800)

    states = []
    probs = []
    rewards = []

    while not game._is_terminal():
        infos = player1.play()
        states.append(infos['state'])
        probs.append(infos['probs'])

        if game._is_terminal():
            break
        infos = player2.play()
        states.append(infos['state'])
        probs.append(infos['probs'])

    winner = game.winner
    for i in range(len(states)):
        current_player = i % 2 + 1

        if current_player == winner:
            rewards.append(1)
        elif winner == 0:
            rewards.append(0)
        else:
            rewards.append(-1)
    
    print(f"Game over! Winner: {winner}")
    game.render()

    return {
        'states': states,
        'probs': probs,
        'rewards': rewards,
    }
    
# %%

board_size = 9
lr = 1e-4
steps = 500
save_per_steps = 100
lab_name = 'gomoku_zero'
batch_size = 128
device = 'cpu'


def train(policy, optimizor, replay_buffer):
    writer = SummaryWriter(f'runs/{lab_name}')

    for step in tqdm.tqdm(range(steps)):
        with torch.no_grad():
            records = self_play(policy)
            for i in range(len(records['states'])):
                replay_buffer.append((
                    records['states'][i],
                    records['probs'][i],
                    records['rewards'][i]
                ))
        
        if len(replay_buffer) < batch_size:
            continue

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
        
        mse = F.mse_loss(value.squeeze(), rewards, reduce='sum')  / batch_size
        cse = -torch.sum(probs * F.log_softmax(logits, dim=-1)) / batch_size

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
    
    buffer = deque(maxlen=20000)

    policy = ZeroPolicy(board_size=board_size, device=device)
    optimizor = torch.optim.Adam(policy.parameters(), lr=lr, weight_decay=1e-4)
    # Create directory if not exists
    train(policy, optimizor, buffer)

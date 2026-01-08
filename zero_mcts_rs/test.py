# 注意：这里 import 的名字就是你文件夹的名字（除非你在 Cargo.toml 改了）
import numpy as np
import torch
import zero_mcts_rs
from gomoku.gomoku_env import GomokuEnv

# 你的 PyTorch policy，返回 (policy_logits, value)
policy = ...  # 例如你的网络


def policy_fn(obs_np: np.ndarray):
    # obs_np: (3, board_size, board_size), float32
    x = torch.from_numpy(obs_np).unsqueeze(0).float()  # (1,3,H,W)
    with torch.no_grad():
        policy_logits, value = policy(x)
    # 返回 numpy array（长度 board_size**2）和 value(float)
    probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
    return probs, float(value.item())


env = GomokuEnv(board_size=9)

mcts = zero_mcts_rs.LightZeroMCTS(
    board_size=9,
    puct=2.0,
    dirichlet_alpha=0.3,
    dirichlet_epsilon=0.25,
)

board_flat = env.board.astype(np.int8).reshape(-1)
action = mcts.run(
    board_flat,
    env.current_player,
    policy_fn,
    iterations=800,
    use_dirichlet=True,
    move_size=env.move_size,
    last_action=env.last_action,
)

# 如果你想要训练时的 pi：
action, pi = mcts.select_action_with_temperature(temperature=1.0, top_k=None)

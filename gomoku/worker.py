
#%%

from gomoku.player import self_play
from gomoku.policy import ZeroPolicy
import ray
import numpy as np

# 1. 初始化 Ray
@ray.remote
class SelfPlayWorker:
    def __init__(self, board_size, device):
        self.policy = ZeroPolicy(board_size=board_size).to(device)
        self.device = device
        self.board_size = board_size
        print(f"Worker actor initialized on {self.device}")

    def set_weights(self, weights):
        self.policy.load_state_dict(weights)
        self.policy.eval()

    def play_game(self, itermax):
        return self_play(self.policy, self.device, itermax)
    
def gather_selfplay_games(policy, device, board_size=9, itermax=800, num_workers=6, games_per_worker=5):
    """
    在单个 worker 上进行 self-play，返回游戏数据。
    """
    policy.eval()

    weights_ref = ray.put({k: v.cpu() for k, v in policy.state_dict().items()})

    workers = [SelfPlayWorker.remote(board_size, device) for _ in range(num_workers)]
    set_weights_tasks = [worker.set_weights.remote(weights_ref) for worker in workers]
    ray.get(set_weights_tasks) # 等待权重设置完成
    game_futures = [worker.play_game.remote(itermax) for _ in range(games_per_worker) for worker in workers] # 简化版任务分配
    games = ray.get(game_futures)
    return games

def get_symmetric_data(state, pi):
    """
    生成棋盘状态和策略的对称增强样本（旋转+翻转）
    假设state形状为(C, 9, 9)，pi长度为81（对应9x9棋盘）
    """
    # 输入校验
    assert isinstance(state, np.ndarray), "state应为numpy数组"
    assert state.ndim == 3 and state.shape[1:] == (9, 9), "state形状应为(C, 9, 9)"
    assert len(pi) == 81, f"pi长度必须为81，实际为{len(pi)}"
    
    pi_board = np.asarray(pi).reshape((9, 9))
    augmented_samples = []
    
    # 4种旋转（0/90/180/270度）+ 水平翻转，共8种对称
    for i in range(4):
        # 旋转
        rotated_state = np.rot90(state, axes=(1, 2), k=i)  # 仅旋转空间维度
        rotated_pi = np.rot90(pi_board, k=i)
        augmented_samples.append((rotated_state, rotated_pi.flatten().tolist()))
        
        # 旋转后翻转
        flipped_state = np.fliplr(rotated_state)
        flipped_pi = np.fliplr(rotated_pi)
        augmented_samples.append((flipped_state, flipped_pi.flatten().tolist()))
    
    return augmented_samples

# 2. 主进程逻辑
if __name__ == '__main__':
    total_games = 9
    iter_max = 40
    board_size = 9
    device = 'cpu'
    num_workers = 6

    ray.init(num_cpus=6)
    policy = ZeroPolicy(board_size=board_size).to(device)
    games = gather_selfplay_games(policy, device, itermax=iter_max)
    ray.shutdown()

    for game in games:
        print(f"Game states: {game['rewards'][0]}")

    print(f"Generated {len(games)} games.")

# %%


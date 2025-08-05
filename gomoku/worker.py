
#%%

from gomoku.player import self_play
from gomoku.policy import ZeroPolicy
import ray

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


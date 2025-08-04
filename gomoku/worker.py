
#%%

from gomoku.trainer import self_play
from gomoku.policy import ZeroPolicy
from collections import deque
import rich

import multiprocessing as mp
import os


def self_play_worker(
    worker_id,
    model_state_dict, # 接收模型权重
    data_queue,       # 用于发送数据的队列
    games_to_play,    # 这个worker需要玩多少局
    board_size,
    itermax,
    device            # 每个worker可以指定自己的device（比如分配不同GPU）
):
    try:  # 捕获所有异常
        policy = ZeroPolicy(board_size=board_size).to(device)
        policy.load_state_dict(model_state_dict)
        policy.eval()

        for i in range(games_to_play):
            records = self_play(policy, device, itermax)
            data_queue.put(records)
            rich.print(f"Worker {worker_id} finished game {i}")
    except Exception as e:
        rich.print(f"Worker {worker_id} error: {e}")
    finally:  # 无论成功失败，都发送'DONE'
        data_queue.put('DONE')
    
    rich.print(f'Worker {worker_id} finished {games_to_play} games')


# %%
total_games = 10
iter_max = 10
board_size = 9
buffer = deque(maxlen=total_games)
device = 'cpu'


if __name__ == '__main__':
    import time
    begin = time.time()
    mp.set_start_method('spawn', force=True)

    # 定义多进程参数
    # num_workers = os.cpu_count() - 1  
    num_workers = 6

    if num_workers <= 0: 
        num_workers = 1
    print(f"Using {num_workers} worker processes for self-play.")

    policy = ZeroPolicy(board_size=board_size).to(device)

    # === 1. 并行生成数据阶段 ===
    policy.eval()

    cpu_model_state_dict = {k: v.cpu() for k, v in policy.state_dict().items()}

    # 创建数据队列
    data_queue = mp.Queue()

    # 计算每个worker需要玩多少局游戏
    games_per_worker = total_games // num_workers
    remaining_games = total_games % num_workers

    processes = []
    for i in range(num_workers):
        games_for_this_worker = games_per_worker + (1 if i < remaining_games else 0)
        if games_for_this_worker == 0: 
            continue

        worker_device = device 

        p = mp.Process(
            target=self_play_worker,
            args=(
                i,
                cpu_model_state_dict,
                data_queue,
                games_for_this_worker,
                board_size, # 你的全局参数
                iter_max,    # 你的全局参数
                worker_device
            )
        )
        p.start()
        processes.append(p)

    games = []

    done_workers = 0

    while done_workers < num_workers:
        game = data_queue.get()
        if game == 'DONE':
            done_workers += 1
            continue
        games.append(game)

    for p in processes:
        p.join()

    print(time.time()-begin)
# %%

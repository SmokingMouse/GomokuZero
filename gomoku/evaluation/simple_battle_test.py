#!/usr/bin/env python3
"""
极简版MCTS+Model vs 普通MCTS 对弈测试
"""

import numpy as np
import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time
import os

def simple_battle(num_games=5, iterations=100):
    """极简对弈测试"""
    print("MCTS+Model vs 普通MCTS 快速测试")
    print("-" * 40)
    
    model_path = 'models/gomoku_zero_9_plus_pro_max/policy_step_199000.pth'
    
    # 加载模型
    policy = ZeroPolicy(board_size=9).to('cpu')
    if os.path.exists(model_path):
        try:
            policy.load_state_dict(torch.load(model_path, map_location='cpu'))
            policy.eval()
            print("✓ 模型加载成功")
        except:
            print("✗ 模型加载失败，使用随机权重")
    else:
        print("✗ 模型文件不存在，使用随机权重")
    
    zero_wins = 0
    total_time = 0
    
    for game in range(num_games):
        print(f"\n第{game+1}局...")
        
        env = GomokuEnv(board_size=9)
        zero_player = ZeroMCTS(env.clone(), policy, device='cpu')
        mcts_player = MCTS(env.clone(), strategy=RandomStrategy(), c=1.41)
        
        # 随机先手
        current_player = "zero" if game % 2 == 0 else "mcts"
        players = {"zero": zero_player, "mcts": mcts_player}
        
        move_count = 0
        start_time = time.time()
        
        while not env._is_terminal() and move_count < 81:
            player = players[current_player]
            action = player.run(iterations=iterations)
            
            env.step(action)
            zero_player.update_root(action)
            
            current_player = "mcts" if current_player == "zero" else "zero"
            move_count += 1
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # 判断胜负
        winner = env.winner
        if winner == 0:
            print("结果：平局")
        elif (winner == 1 and game % 2 == 0) or (winner == 2 and game % 2 == 1):
            print("结果：MCTS+Model获胜")
            zero_wins += 1
        else:
            print("结果：普通MCTS获胜")
    
    # 统计结果
    print(f"\n{'='*40}")
    print("测试完成！")
    print(f"MCTS+Model胜率：{zero_wins}/{num_games} ({zero_wins/num_games:.1%})")
    print(f"平均用时：{total_time/num_games:.2f}秒/局")
    
    return zero_wins / num_games

if __name__ == "__main__":
    simple_battle(num_games=100, iterations=100)
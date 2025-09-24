#!/usr/bin/env python3
"""
轻量级MCTS对弈测试
"""

import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time

def lightweight_battle():
    """超轻量级对弈测试"""
    print("轻量级MCTS对弈测试")
    print("-" * 30)
    
    # 使用小棋盘和很少模拟次数
    board_size = 6
    iterations = 50
    num_games = 3
    
    # 创建环境
    env = GomokuEnv(board_size=board_size)
    
    # 创建MCTS+Model玩家（使用随机权重）
    policy = ZeroPolicy(board_size=board_size).to('cpu')
    zero_player = ZeroMCTS(env.clone(), policy, device='cpu')
    
    # 创建普通MCTS玩家
    mcts_player = MCTS(env.clone(), strategy=RandomStrategy(), c=1.41)
    
    zero_wins = 0
    total_time = 0
    
    for game in range(num_games):
        print(f"\n第{game+1}局 (棋盘{board_size}x{board_size}, 模拟{iterations}次)")
        
        # 重置环境
        env = GomokuEnv(board_size=board_size)
        zero_player = ZeroMCTS(env.clone(), policy, device='cpu')
        
        # 随机先手
        current_player = "zero" if game % 2 == 0 else "mcts"
        players = {"zero": zero_player, "mcts": mcts_player}
        
        move_count = 0
        start_time = time.time()
        
        while not env._is_terminal() and move_count < board_size * board_size:
            try:
                player = players[current_player]
                action = player.run(iterations=iterations)
                
                if action is None:
                    break
                    
                env.step(action)
                zero_player.update_root(action)
                
                current_player = "mcts" if current_player == "zero" else "zero"
                move_count += 1
                
            except Exception as e:
                print(f"游戏出错：{e}")
                break
        
        end_time = time.time()
        game_time = end_time - start_time
        total_time += game_time
        
        # 判断结果
        winner = env.winner
        if winner == 0:
            print("结果：平局")
        elif (winner == 1 and game % 2 == 0) or (winner == 2 and game % 2 == 1):
            print("结果：MCTS+Model获胜")
            zero_wins += 1
        else:
            print("结果：普通MCTS获胜")
        
        print(f"用时：{game_time:.2f}秒，步数：{move_count}")
        
        # 显示最终棋盘
        if move_count > 0:
            print("最终棋盘：")
            env.render()
    
    # 结果统计
    print(f"\n{'='*30}")
    print("测试完成！")
    print(f"MCTS+Model胜率：{zero_wins}/{num_games} ({zero_wins/num_games:.1%})")
    print(f"总用时：{total_time:.2f}秒")
    print(f"平均每局：{total_time/num_games:.2f}秒")
    
    return zero_wins / num_games

if __name__ == "__main__":
    lightweight_battle()
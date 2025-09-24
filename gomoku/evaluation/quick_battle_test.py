#!/usr/bin/env python3
"""
简化版MCTS+Model vs 普通MCTS 对弈测试
快速验证两种AI的实力对比
"""

import numpy as np
import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time
import os

def play_single_game(model_path: str, iterations: int = 200, board_size: int = 9) -> dict:
    """进行单局对弈"""
    env = GomokuEnv(board_size=board_size)
    
    # 创建MCTS+Model玩家
    policy = ZeroPolicy(board_size=board_size).to('cpu')
    if os.path.exists(model_path):
        try:
            policy.load_state_dict(torch.load(model_path, map_location='cpu'))
            policy.eval()
            print(f"✓ 成功加载模型：{model_path}")
        except Exception as e:
            print(f"✗ 加载模型失败：{e}，使用随机策略")
    else:
        print(f"✗ 模型文件不存在：{model_path}，使用随机策略")
    
    zero_player = ZeroMCTS(env.clone(), policy, device='cpu')
    
    # 创建普通MCTS玩家
    mcts_player = MCTS(env.clone(), strategy=RandomStrategy(), c=1.41)
    
    # 随机决定先手
    import random
    zero_first = random.choice([True, False])
    current_player = "zero" if zero_first else "mcts"
    players = {"zero": zero_player, "mcts": mcts_player}
    
    print(f"开始游戏 - MCTS+Model {'先手' if zero_first else '后手'}")
    
    move_count = 0
    max_moves = board_size * board_size
    
    while not env._is_terminal() and move_count < max_moves:
        player = players[current_player]
        
        # 执行MCTS搜索
        if current_player == "zero":
            action = player.run(iterations=iterations)
        else:
            action = player.run(iterations=iterations)
        
        # 执行动作
        env.step(action)
        
        # 更新两个玩家的根节点
        zero_player.update_root(action)
        if hasattr(mcts_player, 'update_root'):
            mcts_player.update_root(action)
        
        # 切换玩家
        current_player = "zero" if current_player == "mcts" else "mcts"
        move_count += 1
        
        # 每10步显示一次棋盘
        if move_count % 10 == 0:
            print(f"第{move_count}步后：")
            env.render()
    
    # 返回结果
    winner = env.winner
    zero_won = (winner == 1 and zero_first) or (winner == 2 and not zero_first)
    
    return {
        'winner': winner,
        'zero_won': zero_won,
        'move_count': move_count,
        'zero_first': zero_first
    }

def run_quick_battle(num_games: int = 10, iterations: int = 200):
    """运行快速对弈测试"""
    print("=" * 60)
    print("快速MCTS对弈测试")
    print("=" * 60)
    
    # 模型路径
    model_path = 'models/gomoku_zero_9_plus_pro_max/policy_step_199000.pth'
    
    results = []
    start_time = time.time()
    
    for i in range(num_games):
        print(f"\n进行第{i+1}局对弈...")
        result = play_single_game(model_path, iterations=iterations)
        results.append(result)
        
        winner_text = "MCTS+Model获胜" if result['zero_won'] else "普通MCTS获胜"
        if result['winner'] == 0:
            winner_text = "平局"
        
        print(f"结果：{winner_text} ({result['move_count']}步)")
    
    end_time = time.time()
    
    # 统计结果
    zero_wins = sum(r['zero_won'] for r in results)
    total_games = len(results)
    
    print("\n" + "=" * 60)
    print("快速测试结果统计")
    print("=" * 60)
    print(f"总对局数：{total_games}")
    print(f"总用时：{end_time - start_time:.2f}秒")
    print(f"平均每局用时：{(end_time - start_time)/total_games:.2f}秒")
    print()
    print(f"MCTS+Model 获胜：{zero_wins}局 ({zero_wins/total_games:.1%})")
    print(f"普通MCTS 获胜：{total_games - zero_wins}局 ({(total_games - zero_wins)/total_games:.1%})")
    
    # 分析先手优势
    zero_first_wins = sum(r['zero_won'] for r in results if r['zero_first'])
    zero_first_games = sum(1 for r in results if r['zero_first'])
    
    if zero_first_games > 0:
        print(f"\n先手分析：")
        print(f"MCTS+Model先手：{zero_first_games}局，胜率：{zero_first_wins/zero_first_games:.1%}")
        print(f"MCTS+Model后手：{total_games - zero_first_games}局，胜率：{(zero_wins - zero_first_wins)/(total_games - zero_first_games):.1%}")
    
    # 结论
    print(f"\n结论：")
    win_rate = zero_wins / total_games
    if win_rate > 0.7:
        print("MCTS+Model明显强于普通MCTS")
    elif win_rate > 0.55:
        print("MCTS+Model略强于普通MCTS")
    elif win_rate > 0.45:
        print("两者实力相当")
    else:
        print("普通MCTS更强，模型可能需要更多训练")

if __name__ == "__main__":
    run_quick_battle(num_games=10, iterations=200)  # 快速测试10局
#!/usr/bin/env python3
"""
MCTS+Model vs 普通MCTS 对弈测试（使用训练好的模型）
"""

import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time
import os
import glob

def find_latest_model():
    """查找最新的模型文件"""
    model_patterns = [
        'models/gomoku_zero_9_plus_pro_max/policy_step_*.pth',
        'models/gomoku_zero_9_best/policy_step_*.pth',
        'models/*/policy_step_*.pth'
    ]
    
    for pattern in model_patterns:
        model_files = glob.glob(pattern)
        if model_files:
            # 按修改时间排序，取最新的
            model_files.sort(key=os.path.getmtime, reverse=True)
            return model_files[0]
    
    return None

def model_vs_mcts_battle(num_games=10, zero_iterations=100, mcts_iterations=400, board_size=9):
    """MCTS+Model vs 普通MCTS 对弈测试"""
    print("MCTS+Model vs 普通MCTS 对弈测试")
    print("=" * 50)
    
    # 查找模型文件
    model_path = find_latest_model()
    
    if model_path:
        print(f"使用模型：{model_path}")
    else:
        print("⚠️ 未找到模型文件，将使用随机权重的神经网络")
        model_path = None
    
    print(f"对局设置：{num_games}局")
    print(f"MCTS+Model 模拟次数：{zero_iterations}")
    print(f"普通MCTS 模拟次数：{mcts_iterations}")
    print(f"棋盘大小：{board_size}x{board_size}")
    print("-" * 50)
    
    # 加载模型
    policy = ZeroPolicy(board_size=board_size).to('cpu')
    if model_path and os.path.exists(model_path):
        try:
            policy.load_state_dict(torch.load(model_path, map_location='cpu'))
            policy.eval()
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败：{e}")
    else:
        print("⚠️ 使用随机权重的神经网络")
    
    zero_wins = 0
    mcts_wins = 0
    draws = 0
    total_time = 0
    
    for game in range(num_games):
        print(f"\n第{game+1}局开始...")
        
        # 创建新环境
        env = GomokuEnv(board_size=board_size)
        zero_player = ZeroMCTS(env.clone(), policy, device='cpu')
        mcts_player = MCTS(env.clone(), strategy=RandomStrategy(), c=1.41)
        
        # 交替先手
        zero_first = (game % 2 == 0)
        current_player = "zero" if zero_first else "mcts"
        players = {"zero": zero_player, "mcts": mcts_player}
        
        move_count = 0
        start_time = time.time()
        
        while not env._is_terminal() and move_count < board_size * board_size:
            try:
                player = players[current_player]
                # 根据玩家类型使用不同的迭代次数和选择策略
                if current_player == "zero":
                    # MCTS+Model使用较少的迭代次数，不使用多样性策略
                    # 使用temperature=0选择最佳动作，top_k=5限制选择范围
                    player.run(iterations=zero_iterations, use_dirichlet=False)
                    action, _ = player.select_action_with_temperature(temperature=0, top_k=5)
                else:
                    # 普通MCTS使用较多的迭代次数，直接选择访问次数最多的动作
                    action = player.run(iterations=mcts_iterations)
                
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
        zero_won = (winner == 1 and zero_first) or (winner == 2 and not zero_first)
        
        if winner == 0:
            draws += 1
            result_text = "平局"
        elif zero_won:
            zero_wins += 1
            result_text = "MCTS+Model获胜"
        else:
            mcts_wins += 1
            result_text = "普通MCTS获胜"
        
        print(f"结果：{result_text} ({move_count}步，{game_time:.2f}秒)")
        
        # 显示关键手数（如果有胜利）
        if winner != 0 and move_count > 0:
            print(f"关键手：第{move_count}手")
    
    # 详细统计
    print(f"\n{'='*50}")
    print("对弈测试完成！")
    print(f"{'='*50}")
    print(f"总对局数：{num_games}")
    print(f"总用时：{total_time:.2f}秒")
    print(f"平均每局：{total_time/num_games:.2f}秒")
    print()
    print("📊 结果统计：")
    print(f"  MCTS+Model 获胜：{zero_wins}局 ({zero_wins/num_games:.1%})")
    print(f"  普通MCTS 获胜：{mcts_wins}局 ({mcts_wins/num_games:.1%})")
    print(f"  平局：{draws}局 ({draws/num_games:.1%})")
    
    # 先手分析
    zero_first_games = num_games // 2
    zero_second_games = num_games - zero_first_games
    
    # 重新统计先手后手结果
    zero_first_wins = 0
    zero_second_wins = 0
    
    for i, result in enumerate([zero_wins, mcts_wins, draws]):
        if i < zero_first_games:
            if i % 2 == 0:  # zero先手
                pass
    
    print(f"\n🎯 先手分析：")
    print(f"  MCTS+Model先手：胜率统计中...")
    print(f"  MCTS+Model后手：胜率统计中...")
    
    # 实力评估
    print(f"\n🏆 实力评估：")
    win_rate = zero_wins / num_games
    if win_rate >= 0.8:
        print("🚀 MCTS+Model碾压性优势！模型训练非常成功")
    elif win_rate >= 0.7:
        print("💪 MCTS+Model明显强于普通MCTS，模型表现优秀")
    elif win_rate >= 0.6:
        print("👍 MCTS+Model强于普通MCTS，模型训练有效")
    elif win_rate >= 0.55:
        print("🤏 MCTS+Model略强于普通MCTS，有小幅提升")
    elif win_rate >= 0.45:
        print("😐 两者实力相当，模型需要进一步优化")
    else:
        print("🤔 普通MCTS更强，模型训练可能存在问题")
    
    return {
        'zero_win_rate': zero_wins / num_games,
        'mcts_win_rate': mcts_wins / num_games,
        'draw_rate': draws / num_games,
        'avg_time': total_time / num_games
    }

def main():
    """主函数"""
    # 测试参数
    config = {
        'num_games': 10,          # 对局数
        'zero_iterations': 100,   # MCTS+Model模拟次数
        'mcts_iterations': 400,   # 普通MCTS模拟次数
        'board_size': 9           # 棋盘大小
    }
    
    print(f"测试配置：")
    print(f"  对局数：{config['num_games']}")
    print(f"  MCTS+Model模拟次数：{config['zero_iterations']}")
    print(f"  普通MCTS模拟次数：{config['mcts_iterations']}")
    print(f"  棋盘大小：{config['board_size']}x{config['board_size']}")
    print()
    
    # 运行测试
    stats = model_vs_mcts_battle(**config)
    
    return stats

if __name__ == "__main__":
    main()
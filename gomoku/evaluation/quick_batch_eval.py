#!/usr/bin/env python3
"""
快速批量模型评估
评估关键训练节点的模型性能
"""

import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time
import os

def quick_evaluate_model(model_path, num_games=20, zero_iterations=100, mcts_iterations=400):
    """快速评估单个模型"""
    print(f"评估：{os.path.basename(model_path)}")
    
    # 加载模型
    policy = ZeroPolicy(board_size=9).to('cpu')
    try:
        policy.load_state_dict(torch.load(model_path, map_location='cpu'))
        policy.eval()
    except Exception as e:
        print(f"模型加载失败：{e}")
        return None
    
    zero_wins = 0
    total_time = 0
    
    for game in range(num_games):
        if game % 5 == 0 and game > 0:
            print(f"  进度：{game}/{num_games}")
        
        env = GomokuEnv(board_size=9)
        zero_player = ZeroMCTS(env.clone(), policy, device='cpu')
        mcts_player = MCTS(env.clone(), strategy=RandomStrategy(), c=1.41)
        
        zero_first = (game % 2 == 0)
        current_player = "zero" if zero_first else "mcts"
        players = {"zero": zero_player, "mcts": mcts_player}
        
        move_count = 0
        start_time = time.time()
        
        while not env._is_terminal() and move_count < 81:
            player = players[current_player]
            if current_player == "zero":
                player.run(iterations=zero_iterations, use_dirichlet=False)
                action, _ = player.select_action_with_temperature(temperature=0, top_k=5)
            else:
                action = player.run(iterations=mcts_iterations)
            
            if action is None:
                break
                
            env.step(action)
            zero_player.update_root(action)
            
            current_player = "mcts" if current_player == "zero" else "zero"
            move_count += 1
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        # 判断结果
        winner = env.winner
        zero_won = (winner == 1 and zero_first) or (winner == 2 and not zero_first)
        
        if winner == 0:
            pass  # 平局
        elif zero_won:
            zero_wins += 1
    
    win_rate = zero_wins / num_games
    avg_time = total_time / num_games
    
    print(f"  结果：胜率 {win_rate:.1%} ({zero_wins}/{num_games})")
    print(f"  平均用时：{avg_time:.2f}秒/局")
    
    return {
        'model': os.path.basename(model_path),
        'win_rate': win_rate,
        'wins': zero_wins,
        'avg_time': avg_time
    }

def quick_batch_eval():
    """快速批量评估关键模型"""
    print("快速批量模型评估")
    print("=" * 50)
    
    # 选择关键评估点
    key_models = [f'../{elem}' for elem in [
        "models/gomoku_zero_9_pre/policy_step_50000.pth",   # 1万步
        # 'models/gomoku_zero_9_plus_pro_max/policy_step_10000.pth',   # 1万步
        # 'models/gomoku_zero_9_plus_pro_max/policy_step_50000.pth',   # 5万步
        # 'models/gomoku_zero_9_plus_pro_max/policy_step_100000.pth',  # 10万步
        # 'models/gomoku_zero_9_plus_pro_max/policy_step_150000.pth',  # 15万步
        # 'models/gomoku_zero_9_plus_pro_max/policy_step_199000.pth',  # 19.9万步
    ]]
    print(key_models)
    
    results = []
    total_start = time.time()
    
    for i, model_path in enumerate(key_models):
        print(f"\n[{i+1}/{len(key_models)}] ", end="")
        
        if not os.path.exists(model_path):
            print(f"文件不存在：{model_path}")
            continue
            
        result = quick_evaluate_model(model_path, num_games=20)
        if result:
            results.append(result)
    
    total_end = time.time()
    
    # 显示结果
    print(f"\n{'='*50}")
    print("评估完成！")
    print(f"总用时：{(total_end - total_start)/60:.1f}分钟")
    print(f"{'='*50}")
    
    print("\n模型性能对比：")
    print("-" * 50)
    print(f"{'模型':<25} {'胜率':<8} {'胜场':<8} {'平均用时'}")
    print("-" * 50)
    
    for result in results:
        step = result['model'].split('step_')[1].split('.pth')[0]
        print(f"Step {step:<20} {result['win_rate']:.1%}{'':<4} {result['wins']:<8} {result['avg_time']:.2f}秒")
    
    # 找出最佳模型
    if results:
        best_model = max(results, key=lambda x: x['win_rate'])
        print(f"\n🏆 最佳模型：{best_model['model']}")
        print(f"   胜率：{best_model['win_rate']:.1%}")
        
        # 趋势分析
        win_rates = [r['win_rate'] for r in results]
        if len(win_rates) >= 3:
            early_avg = sum(win_rates[:2]) / 2
            late_avg = sum(win_rates[-2:]) / 2
            
            print(f"\n📈 训练趋势：")
            print(f"   早期平均胜率：{early_avg:.1%}")
            print(f"   后期平均胜率：{late_avg:.1%}")
            
            if late_avg > early_avg:
                print("   ✅ 模型性能随训练提升")
            else:
                print("   ⚠️ 性能提升有限")
    
    return results

if __name__ == "__main__":
    quick_batch_eval()
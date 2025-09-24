#!/usr/bin/env python3
"""
批量模型评估脚本
每10000步评估一次模型性能
"""

import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time
import os
import glob
import json
from datetime import datetime
import numpy as np

def find_models_by_step(model_dir, step_interval=10000):
    """查找指定步数间隔的模型文件"""
    pattern = os.path.join(model_dir, "policy_step_*.pth")
    model_files = glob.glob(pattern)
    
    models = []
    for model_file in model_files:
        # 提取步数
        filename = os.path.basename(model_file)
        try:
            step = int(filename.split("policy_step_")[1].split(".pth")[0])
            if step % step_interval == 0:
                models.append({
                    'path': model_file,
                    'step': step,
                    'filename': filename
                })
        except:
            continue
    
    # 按步数排序
    models.sort(key=lambda x: x['step'])
    return models

def evaluate_single_model(model_path, num_games=40, zero_iterations=100, mcts_iterations=400, board_size=9):
    """评估单个模型"""
    print(f"\n评估模型：{os.path.basename(model_path)}")
    print("-" * 50)
    
    # 加载模型
    policy = ZeroPolicy(board_size=board_size).to('cpu')
    try:
        if os.path.exists(model_path):
            policy.load_state_dict(torch.load(model_path, map_location='cpu'))
            policy.eval()
            print("✅ 模型加载成功")
        else:
            print(f"❌ 模型文件不存在：{model_path}")
            return None
    except Exception as e:
        print(f"❌ 模型加载失败：{e}")
        return None
    
    zero_wins = 0
    mcts_wins = 0
    draws = 0
    total_time = 0
    game_details = []
    
    for game in range(num_games):
        if game % 10 == 0 and game > 0:
            print(f"进度：{game}/{num_games}")
        
        # 创建环境
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
                if current_player == "zero":
                    # MCTS+Model：使用最佳策略，无多样性，top_k=5
                    player.run(iterations=zero_iterations, use_dirichlet=False)
                    action, _ = player.select_action_with_temperature(temperature=0, top_k=5)
                else:
                    # 普通MCTS：使用更多迭代
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
            result = "draw"
        elif zero_won:
            zero_wins += 1
            result = "zero_win"
        else:
            mcts_wins += 1
            result = "mcts_win"
        
        game_details.append({
            'game': game + 1,
            'result': result,
            'zero_first': zero_first,
            'move_count': move_count,
            'time': game_time
        })
    
    # 计算统计
    win_rate = zero_wins / num_games if num_games > 0 else 0
    avg_time = total_time / num_games if num_games > 0 else 0
    avg_moves = np.mean([g['move_count'] for g in game_details]) if game_details else 0
    
    # 先手后手分析
    zero_first_games = [g for g in game_details if g['zero_first']]
    zero_second_games = [g for g in game_details if not g['zero_first']]
    
    zero_first_wins = sum(1 for g in zero_first_games if g['result'] == 'zero_win')
    zero_second_wins = sum(1 for g in zero_second_games if g['result'] == 'zero_win')
    
    zero_first_rate = zero_first_wins / len(zero_first_games) if zero_first_games else 0
    zero_second_rate = zero_second_wins / len(zero_second_games) if zero_second_games else 0
    
    return {
        'model_path': model_path,
        'zero_wins': zero_wins,
        'mcts_wins': mcts_wins,
        'draws': draws,
        'total_games': num_games,
        'win_rate': win_rate,
        'avg_time': avg_time,
        'avg_moves': avg_moves,
        'zero_first_rate': zero_first_rate,
        'zero_second_rate': zero_second_rate,
        'game_details': game_details
    }

def batch_evaluate_models(model_dir, output_file=None, step_interval=10000, num_games=40):
    """批量评估模型"""
    print("批量模型评估")
    print("=" * 60)
    print(f"模型目录：{model_dir}")
    print(f"评估间隔：每{step_interval}步")
    print(f"每模型对局数：{num_games}")
    print("-" * 60)
    
    # 查找模型
    models = find_models_by_step(model_dir, step_interval)
    
    if not models:
        print(f"未找到符合要求的模型文件（{model_dir}/policy_step_*.pth，间隔{step_interval}）")
        return
    
    print(f"找到{len(models)}个模型：")
    for model in models:
        print(f"  Step {model['step']:6d}: {model['filename']}")
    print()
    
    # 评估每个模型
    results = []
    total_start_time = time.time()
    
    for i, model_info in enumerate(models):
        print(f"\n[{i+1}/{len(models)}] 评估Step {model_info['step']}的模型...")
        
        result = evaluate_single_model(
            model_path=model_info['path'],
            num_games=num_games,
            zero_iterations=100,
            mcts_iterations=400,
            board_size=9
        )
        
        if result:
            result['step'] = model_info['step']
            results.append(result)
            
            # 实时显示结果
            print(f"结果：胜率{result['win_rate']:.1%} ({result['zero_wins']}/{num_games})")
            print(f"      平均用时：{result['avg_time']:.2f}秒/局")
            
            # 先手后手分析
            if result['zero_first_rate'] > 0 or result['zero_second_rate'] > 0:
                print(f"      先手胜率：{result['zero_first_rate']:.1%}, 后手胜率：{result['zero_second_rate']:.1%}")
    
    total_end_time = time.time()
    
    # 生成报告
    print(f"\n{'='*60}")
    print("评估完成！")
    print(f"总用时：{(total_end_time - total_start_time)/60:.1f}分钟")
    print(f"{'='*60}")
    
    # 按胜率排序
    results.sort(key=lambda x: x['win_rate'], reverse=True)
    
    print("\n模型性能排行榜：")
    print("-" * 60)
    print(f"{'排名':<4} {'Step':<8} {'胜率':<8} {'胜场':<8} {'平均用时':<10} {'先手胜率':<10} {'后手胜率'}")
    print("-" * 60)
    
    for i, result in enumerate(results):
        print(f"{i+1:<4} {result['step']:<8} {result['win_rate']:.1%}{'':<4} {result['zero_wins']:<8} "
              f"{result['avg_time']:<10.2f} {result['zero_first_rate']:.1%}{'':<6} {result['zero_second_rate']:.1%}")
    
    # 找出最佳模型
    if results:
        best_model = results[0]
        print(f"\n🏆 最佳模型：Step {best_model['step']}")
        print(f"   胜率：{best_model['win_rate']:.1%}")
        print(f"   模型文件：{os.path.basename(best_model['model_path'])}")
        
        # 分析训练趋势
        if len(results) > 1:
            print(f"\n📈 训练趋势分析：")
            early_models = results[:len(results)//2]
            late_models = results[len(results)//2:]
            
            early_avg_winrate = np.mean([r['win_rate'] for r in early_models])
            late_avg_winrate = np.mean([r['win_rate'] for r in late_models])
            
            print(f"   早期模型平均胜率：{early_avg_winrate:.1%}")
            print(f"   后期模型平均胜率：{late_avg_winrate:.1%}")
            
            if late_avg_winrate > early_avg_winrate:
                print("   ✅ 模型性能随训练提升")
            elif late_avg_winrate < early_avg_winrate:
                print("   ⚠️ 模型性能随训练下降，可能需要调整训练参数")
            else:
                print("   🤔 模型性能变化不大，可能已达到瓶颈")
    
    # 保存结果到文件
    if output_file:
        save_results(results, output_file)
        print(f"\n💾 详细结果已保存到：{output_file}")
    
    return results

def save_results(results, output_file):
    """保存结果到JSON文件"""
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'total_models': len(results),
        'summary': {
            'best_model_step': results[0]['step'] if results else None,
            'best_win_rate': results[0]['win_rate'] if results else 0,
            'average_win_rate': np.mean([r['win_rate'] for r in results]) if results else 0
        },
        'results': results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

def main():
    """主函数"""
    # 配置
    config = {
        'model_dir': 'models/gomoku_zero_9_plus_pro_max',  # 模型目录
        'step_interval': 10000,  # 每10000步评估一次
        'num_games': 40,         # 每模型对局数
        'output_file': f'model_evaluation_{int(time.time())}.json'  # 输出文件名
    }
    
    print("模型批量评估工具")
    print("=" * 60)
    print(f"目标目录：{config['model_dir']}")
    print(f"评估间隔：每{config['step_interval']}步")
    print(f"每模型对局：{config['num_games']}局")
    print(f"输出文件：{config['output_file']}")
    print("-" * 60)
    
    # 运行批量评估
    results = batch_evaluate_models(**config)
    
    if results:
        print(f"\n✅ 批量评估完成，共评估了{len(results)}个模型")
    else:
        print("\n❌ 没有找到符合条件的模型")

if __name__ == "__main__":
    main()
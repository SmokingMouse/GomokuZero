#!/usr/bin/env python3
"""
MCTS+Model vs 普通MCTS 对弈测试脚本
通过胜率来衡量AI水平
"""

import numpy as np
import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy
from gomoku.policy import ZeroPolicy
import time
import concurrent.futures
from typing import List, Tuple
import os

class BattleTester:
    def __init__(self, board_size=9, num_games=100, iterations=800):
        self.board_size = board_size
        self.num_games = num_games
        self.iterations = iterations
        self.results = []
        
    def create_mcts_player(self, env: GomokuEnv) -> MCTS:
        """创建普通MCTS玩家（无神经网络）"""
        return MCTS(env, strategy=RandomStrategy(), c=1.41)
    
    def create_zero_mcts_player(self, env: GomokuEnv, model_path: str) -> ZeroMCTS:
        """创建MCTS+Model玩家"""
        policy = ZeroPolicy(board_size=self.board_size).to('cpu')
        
        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            print(f"警告：模型文件 {model_path} 不存在，使用随机策略")
        else:
            try:
                policy.load_state_dict(torch.load(model_path, map_location='cpu'))
                policy.eval()
                print(f"成功加载模型：{model_path}")
            except Exception as e:
                print(f"加载模型失败：{e}，使用随机策略")
        
        return ZeroMCTS(env.clone(), policy, device='cpu')
    
    def play_single_game(self, game_idx: int, zero_first: bool, model_path: str) -> dict:
        """进行单局对弈"""
        env = GomokuEnv(board_size=self.board_size)
        
        # 创建玩家
        zero_player = self.create_zero_mcts_player(env, model_path)
        mcts_player = self.create_mcts_player(env)
        
        # 设定先手
        if zero_first:
            current_player = "zero"
            players = {"zero": zero_player, "mcts": mcts_player}
        else:
            current_player = "mcts" 
            players = {"mcts": mcts_player, "zero": zero_player}
        
        move_count = 0
        max_moves = self.board_size * self.board_size
        
        while not env._is_terminal() and move_count < max_moves:
            player = players[current_player]
            
            if current_player == "zero":
                # MCTS+Model玩家
                action = player.run(iterations=self.iterations)
            else:
                # 普通MCTS玩家
                action = player.run(iterations=self.iterations)
            
            # 执行动作
            env.step(action)
            
            # 更新两个玩家的根节点
            zero_player.update_root(action)
            mcts_player.update_root(action) if hasattr(mcts_player, 'update_root') else None
            
            # 切换玩家
            current_player = "zero" if current_player == "mcts" else "mcts"
            move_count += 1
        
        # 返回结果
        winner = env.winner
        return {
            'game_idx': game_idx,
            'zero_first': zero_first,
            'winner': winner,
            'move_count': move_count,
            'zero_wins': 1 if (winner == 1 and zero_first) or (winner == 2 and not zero_first) else 0,
            'mcts_wins': 1 if (winner == 2 and zero_first) or (winner == 1 and not zero_first) else 0,
            'draws': 1 if winner == 0 else 0
        }
    
    def run_battle(self, model_path: str, num_workers: int = 4) -> dict:
        """运行对弈测试"""
        print(f"开始MCTS+Model vs 普通MCTS对弈测试")
        print(f"总对局数：{self.num_games}")
        print(f"每局模拟次数：{self.iterations}")
        print(f"模型路径：{model_path}")
        print("-" * 50)
        
        start_time = time.time()
        
        # 使用多进程加速对弈
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            # 提交任务，交替先手
            futures = []
            for i in range(self.num_games):
                zero_first = (i % 2 == 0)  # 交替先手
                future = executor.submit(self.play_single_game, i, zero_first, model_path)
                futures.append(future)
            
            # 收集结果
            self.results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    self.results.append(result)
                    if len(self.results) % 10 == 0:
                        print(f"完成对局：{len(self.results)}/{self.num_games}")
                except Exception as e:
                    print(f"对局执行失败：{e}")
        
        end_time = time.time()
        
        # 统计结果
        stats = self.calculate_stats()
        stats['total_time'] = end_time - start_time
        
        return stats
    
    def calculate_stats(self) -> dict:
        """计算统计信息"""
        if not self.results:
            return {}
        
        total_zero_wins = sum(r['zero_wins'] for r in self.results)
        total_mcts_wins = sum(r['mcts_wins'] for r in self.results)
        total_draws = sum(r['draws'] for r in self.results)
        
        # 分别统计先手和后手的表现
        zero_first_games = [r for r in self.results if r['zero_first']]
        zero_second_games = [r for r in self.results if not r['zero_first']]
        
        zero_first_wins = sum(r['zero_wins'] for r in zero_first_games)
        zero_second_wins = sum(r['zero_wins'] for r in zero_second_games)
        
        return {
            'total_games': len(self.results),
            'zero_wins': total_zero_wins,
            'mcts_wins': total_mcts_wins,
            'draws': total_draws,
            'zero_win_rate': total_zero_wins / len(self.results),
            'mcts_win_rate': total_mcts_wins / len(self.results),
            'draw_rate': total_draws / len(self.results),
            'zero_first_games': len(zero_first_games),
            'zero_second_games': len(zero_second_games),
            'zero_first_win_rate': zero_first_wins / len(zero_first_games) if zero_first_games else 0,
            'zero_second_win_rate': zero_second_wins / len(zero_second_games) if zero_second_games else 0,
            'avg_moves': np.mean([r['move_count'] for r in self.results])
        }
    
    def print_stats(self, stats: dict):
        """打印统计结果"""
        print("\n" + "=" * 60)
        print("对弈测试结果统计")
        print("=" * 60)
        print(f"总对局数：{stats['total_games']}")
        print(f"总用时：{stats['total_time']:.2f}秒")
        print(f"平均每局用时：{stats['total_time']/stats['total_games']:.2f}秒")
        print(f"平均步数：{stats['avg_moves']:.1f}")
        print()
        print("MCTS+Model vs 普通MCTS：")
        print(f"  MCTS+Model 胜率：{stats['zero_win_rate']:.1%} ({stats['zero_wins']}/{stats['total_games']})")
        print(f"  普通MCTS 胜率：{stats['mcts_win_rate']:.1%} ({stats['mcts_wins']}/{stats['total_games']})")
        print(f"  平局率：{stats['draw_rate']:.1%} ({stats['draws']}/{stats['total_games']})")
        print()
        print("先手/后手分析：")
        print(f"  MCTS+Model先手：{stats['zero_first_games']}局，胜率{stats['zero_first_win_rate']:.1%}")
        print(f"  MCTS+Model后手：{stats['zero_second_games']}局，胜率{stats['zero_second_win_rate']:.1%}")
        
        # 判断AI水平
        print()
        if stats['zero_win_rate'] > 0.7:
            print("结论：MCTS+Model明显强于普通MCTS")
        elif stats['zero_win_rate'] > 0.55:
            print("结论：MCTS+Model略强于普通MCTS")
        elif stats['zero_win_rate'] > 0.45:
            print("结论：两者实力相当")
        else:
            print("结论：普通MCTS更强，模型可能需要更多训练")

def main():
    # 测试参数
    BOARD_SIZE = 9
    NUM_GAMES = 100  # 对局数
    ITERATIONS = 800  # 每局MCTS模拟次数
    NUM_WORKERS = 4  # 并行进程数
    
    # 模型路径列表（可以测试多个模型）
    model_paths = [
        'models/gomoku_zero_9_plus_pro_max/policy_step_199000.pth',
        'models/gomoku_zero_9_best/policy_step_100000.pth',
        # 添加更多模型路径...
    ]
    
    # 使用第一个存在的模型
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("警告：未找到任何模型文件，将创建随机策略的MCTS+Model进行对比")
        model_path = "random_model.pth"  # 不存在的文件，会回退到随机策略
    
    # 创建测试器
    tester = BattleTester(
        board_size=BOARD_SIZE,
        num_games=NUM_GAMES,
        iterations=ITERATIONS
    )
    
    # 运行测试
    stats = tester.run_battle(model_path, num_workers=NUM_WORKERS)
    
    # 打印结果
    tester.print_stats(stats)

if __name__ == "__main__":
    main()
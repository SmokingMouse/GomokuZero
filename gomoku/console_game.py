#!/usr/bin/env python3
"""
控制台版五子棋游戏
适用于没有图形界面的环境
"""

import sys
import os
from gomoku.gomoku_env import GomokuEnv
from gomoku.mcts import MCTS


class ConsoleGomoku:
    def __init__(self, board_size=15):
        self.board_size = board_size
        self.env = GomokuEnv(board_size)
        self.ai = MCTS(strategy="random", c=1.41)
        
    def clear_screen(self):
        """Clear the console screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_board(self):
        """Print the game board"""
        self.clear_screen()
        print("\n" + "="*50)
        print("          五子棋游戏")
        print("="*50)
        
        # Print column numbers
        print("   ", end="")
        for j in range(self.board_size):
            print(f"{j:2d}", end=" ")
        print()
        
        # Print board
        symbols = {0: '.', 1: '●', 2: '○'}
        for i, row in enumerate(self.env.board):
            print(f"{i:2d} ", end="")
            for cell in row:
                print(f"{symbols[cell]} ", end=" ")
            print()
        
        print("\n" + "-"*50)
        print(f"当前玩家: {'你 (●)' if self.env.current_player == 1 else 'AI (○)'}")
        
        if self.env.winner:
            if self.env.winner == 1:
                print("🎉 你赢了！")
            elif self.env.winner == 2:
                print("🤖 AI赢了！")
            else:
                print("🤝 平局！")
        print("-"*50)
    
    def get_human_move(self):
        """Get move from human player"""
        valid_actions = set(self.env.get_valid_actions())
        
        if not valid_actions:
            return None
            
        while True:
            try:
                print("\n请输入你的落子位置 (格式: 行 列，例如: 7 7)")
                print("或输入 'q' 退出, 'r' 重新开始")
                user_input = input("> ").strip().lower()
                
                if user_input in ['q', 'quit', 'exit']:
                    return 'quit'
                elif user_input in ['r', 'restart']:
                    return 'restart'
                elif user_input in ['h', 'help']:
                    self.print_help()
                    continue
                
                # Parse coordinates
                parts = user_input.split()
                if len(parts) != 2:
                    print("❌ 请输入两个数字，格式: 行 列")
                    continue
                
                row, col = int(parts[0]), int(parts[1])
                
                if not (0 <= row < self.board_size and 0 <= col < self.board_size):
                    print(f"❌ 坐标必须在 0-{self.board_size-1} 之间")
                    continue
                
                action = row * self.board_size + col
                
                if action not in valid_actions:
                    print("❌ 该位置已经被占用")
                    continue
                
                return action
                
            except ValueError:
                print("❌ 请输入有效的数字")
            except KeyboardInterrupt:
                print("\n👋 游戏结束")
                return 'quit'
    
    def print_help(self):
        """Print game help"""
        self.clear_screen()
        print("\n" + "="*50)
        print("          游戏帮助")
        print("="*50)
        print("● 游戏目标：在横、竖或斜方向上连成5子")
        print("● 落子方法：输入行列坐标，如 '7 7' 表示第7行第7列")
        print("● 游戏控制：")
        print("  - 'q' 或 'quit' 退出游戏")
        print("  - 'r' 或 'restart' 重新开始")
        print("  - 'h' 或 'help' 查看帮助")
        print("="*50)
        input("按回车键继续...")
    
    def ai_move(self):
        """AI makes a move"""
        print("🤖 AI正在思考中...")
        
        # iterations = min(6000, max(100, len(self.env.get_valid_actions()) * 31))
        iterations = 8000
        action = self.ai.run(self.env, iterations)
        
        row, col = action // self.board_size, action % self.board_size
        print(f"🤖 AI落子: {row} {col}")
        
        return action
    
    def play_game(self):
        """Main game loop"""
        print("🎮 欢迎来到五子棋游戏！")
        print("🎯 你是黑子(●)，AI是白子(○)")
        print("📖 输入 'h' 查看帮助")
        
        while True:
            self.print_board()
            
            if self.env._is_terminal():
                choice = input("\n再来一局? (y/n): ").strip().lower()
                if choice in ['y', 'yes', '']:
                    self.env.reset()
                    continue
                else:
                    break
            
            if self.env.current_player == 1:
                # Human turn
                action = self.get_human_move()
                
                if action == 'quit':
                    print("👋 感谢游戏！")
                    break
                elif action == 'restart':
                    self.env.reset()
                    print("🔄 游戏重新开始")
                    continue
                elif action is None:
                    continue
                    
            else:
                # AI turn
                action = self.ai_move()
            
            # Make the move
            self.env.step(action)
    
    def run(self):
        """Start the game"""
        try:
            self.play_game()
        except KeyboardInterrupt:
            print("\n👋 游戏结束")


def main():
    print("请选择游戏模式:")
    print("1. 标准棋盘 (15×15)")
    print("2. 小棋盘 (9×9)")
    
    try:
        choice = input("请选择 (1/2): ").strip()
        if choice == "2":
            board_size = 9
        else:
            board_size = 15
    except KeyboardInterrupt:
        print("\n游戏取消")
        return
    
    game = ConsoleGomoku(board_size)
    game.run()


if __name__ == "__main__":
    main()
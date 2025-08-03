#!/usr/bin/env python3
"""
控制台版五子棋游戏启动器
适用于没有图形界面的环境
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gomoku.console_game import ConsoleGomoku

if __name__ == "__main__":
    print("启动控制台版五子棋游戏...")
    print("=" * 50)
    print("游戏控制：")
    print("- 输入 '行 列' 落子，例如：7 7")
    print("- 输入 'q' 退出")
    print("- 输入 'r' 重新开始")
    print("- 输入 'h' 查看帮助")
    print("=" * 50)
    
    try:
        game = ConsoleGomoku()
        game.run()
    except KeyboardInterrupt:
        print("\n游戏结束")
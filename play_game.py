#!/usr/bin/env python3
"""
交互式五子棋游戏启动器
运行此脚本开始游戏
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gomoku.interactive_game import GomokuGUI

if __name__ == "__main__":
    print("启动五子棋游戏...")
    print("你可以通过鼠标点击来下棋")
    print("黑子(●)是你，白子(○)是AI")
    
    # 创建并运行游戏
    game = GomokuGUI(board_size=15, cell_size=30)
    game.run()
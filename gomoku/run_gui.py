#!/usr/bin/env python3
"""
Gomoku AI Battle Arena 启动脚本
运行这个脚本启动对战界面
"""

from gui_new import GomokuBattleGUI

def main():
    print("正在启动 Gomoku AI Battle Arena...")
    print("=" * 50)
    print("使用说明:")
    print("1. 点击棋盘进行下棋")
    print("2. 空格键: 切换游戏模式 (人对AI, AI对AI, 人对人)")
    print("3. R键: 重新开始游戏")
    print("4. S键: 打开设置对话框")
    print("=" * 50)

    game = GomokuBattleGUI()
    game.run()

if __name__ == "__main__":
    main()
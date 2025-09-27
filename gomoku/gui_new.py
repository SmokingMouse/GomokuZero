import pygame
import tkinter as tk
from tkinter import ttk
import threading
import time
import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS, RandomStrategy

# 游戏常量
BOARD_SIZE = 9
SQUARE_SIZE = 40
MARGIN = 40
WIDTH = HEIGHT = SQUARE_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (210, 180, 140)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

class GomokuBattleGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH + 300, HEIGHT))  # 增加右侧控制面板
        pygame.display.set_caption("Gomoku AI Battle Arena")
        self.clock = pygame.time.Clock()

        try:
            # 定义一个包含多个中文字体名称的列表，按优先级排序
            # 'WenQuanYi Zen Hei' 是在 Ubuntu/WSL 中安装的字体
            # 'Microsoft YaHei UI' 等作为在 Windows 原生环境运行时的备选
            font_names = "WenQuanYi Zen Hei, Microsoft YaHei UI, SimHei, SimSun"
            
            print(f"尝试从系统字体库加载: {font_names}")
            
            # 使用 SysFont 来加载字体。它会按顺序尝试列表中的字体。
            # 如果都找不到，它会自动回退到 Pygame 的一个默认字体，而不会报错。
            self.font = pygame.font.SysFont(font_names, 32)
            self.small_font = pygame.font.SysFont(font_names, 20)
            
            # 由于无法直接查询加载了哪个字体，我们通过打印消息来提示用户进行视觉确认
            print("字体对象创建成功。请通过界面显示效果，确认中文字体是否加载正确。")

        except Exception as e:
            # 这个 except 块主要用于捕捉 Pygame 字体系统初始化失败等更严重的问题
            print(f"加载系统字体时出现严重错误: {e}, 将使用 Pygame 默认字体 (可能乱码)")
            self.font = pygame.font.Font(None, 32)
            self.small_font = pygame.font.Font(None, 20)

        # 游戏环境
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game_mode = "human_vs_ai"  # human_vs_ai, ai_vs_ai, human_vs_human
        self.game_over = False
        self.current_player = 1  # 1 for black, 2 for white
        self.winner = None

        # AI设置
        self.zero_iterations = 400
        self.ai_iterations = 4000
        self.ai_temperature = 0.1
        self.device = 'cpu'

        # 初始化AI
        self._init_ai_players()

        # 人类玩家设置
        self.human_player = 2  # 默认人类执黑

        # 后台搜索
        self.searching = False
        self.search_thread = None
        self.ai_thinking = False

        # 控制面板
        self.show_control_panel = True

    def _init_ai_players(self):
        """初始化AI玩家"""
        # ZeroMCTS AI (使用神经网络)
        try:
            self.zero_policy = ZeroPolicy(board_size=BOARD_SIZE, num_blocks=2).to(self.device)
            self.zero_policy.load_state_dict(torch.load('models/gomoku_zero_9_pre2/policy_step_660000.pth', map_location=self.device))
            self.zero_policy.eval()
            self.zero_mcts_player = ZeroMCTS(self.env.clone(), self.zero_policy, device=self.device)
        except Exception as e:
            print(f"Warning: Could not load ZeroMCTS model ({e}), using pure MCTS instead")
            self.zero_mcts_player = None

        # 纯MCTS AI
        self.pure_mcts_player = MCTS(self.env.clone(), RandomStrategy(), c=1.41)

    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(BOARD_COLOR)

        # 绘制网格线
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, BLACK,
                           (MARGIN, MARGIN + i * SQUARE_SIZE),
                           (WIDTH - MARGIN, MARGIN + i * SQUARE_SIZE), 2)
            pygame.draw.line(self.screen, BLACK,
                           (MARGIN + i * SQUARE_SIZE, MARGIN),
                           (MARGIN + i * SQUARE_SIZE, HEIGHT - MARGIN), 2)

        # 绘制星位
        star_points = [(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)]
        for r, c in star_points:
            pygame.draw.circle(self.screen, BLACK,
                             (MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE), 4)

    def draw_pieces(self):
        """绘制棋子"""
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.env.board[r, c] == 1:  # Black
                    pygame.draw.circle(self.screen, BLACK,
                                     (MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE),
                                     SQUARE_SIZE // 2 - 2)
                elif self.env.board[r, c] == 2:  # White
                    pygame.draw.circle(self.screen, WHITE,
                                     (MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE),
                                     SQUARE_SIZE // 2 - 2)
                    pygame.draw.circle(self.screen, BLACK,
                                     (MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE),
                                     SQUARE_SIZE // 2 - 2, 1)

    def draw_control_panel(self):
        """绘制右侧控制面板"""
        panel_x = WIDTH + 10
        panel_y = 10

        # 绘制面板背景
        pygame.draw.rect(self.screen, (240, 240, 240), (WIDTH, 0, 300, HEIGHT))
        pygame.draw.line(self.screen, BLACK, (WIDTH, 0), (WIDTH, HEIGHT), 2)

        # 标题
        title = self.font.render("控制面板", True, BLACK)
        self.screen.blit(title, (panel_x, panel_y))
        panel_y += 40

        # 游戏模式
        mode_text = self.small_font.render(f"模式: {self.game_mode}", True, BLACK)
        self.screen.blit(mode_text, (panel_x, panel_y))
        panel_y += 30

        # 当前玩家
        player_text = self.small_font.render(f"当前: {'黑棋' if self.current_player == 1 else '白棋'}", True, BLACK)
        self.screen.blit(player_text, (panel_x, panel_y))
        panel_y += 30

        # AI设置
        if "ai" in self.game_mode:
            ai_text = self.small_font.render(f"AI迭代: {self.ai_iterations}", True, BLACK)
            self.screen.blit(ai_text, (panel_x, panel_y))
            panel_y += 20

            temp_text = self.small_font.render(f"温度: {self.ai_temperature}", True, BLACK)
            self.screen.blit(temp_text, (panel_x, panel_y))
            panel_y += 30

        # 游戏状态
        if self.game_over:
            if self.winner == 0:
                status_text = "游戏结束: 平局"
            else:
                status_text = f"游戏结束: {'黑棋' if self.winner == 1 else '白棋'}获胜"
            status_color = RED
        else:
            if self.ai_thinking:
                status_text = "AI思考中..."
                status_color = BLUE
            else:
                status_text = "游戏进行中"
                status_color = GREEN

        status = self.small_font.render(status_text, True, status_color)
        self.screen.blit(status, (panel_x, panel_y))
        panel_y += 40

        # 操作说明
        self.screen.blit(self.small_font.render("操作说明:", True, BLACK), (panel_x, panel_y))
        panel_y += 20
        self.screen.blit(self.small_font.render("点击棋盘: 下棋", True, BLACK), (panel_x, panel_y))
        panel_y += 20
        self.screen.blit(self.small_font.render("空格键: 切换模式", True, BLACK), (panel_x, panel_y))
        panel_y += 20
        self.screen.blit(self.small_font.render("R键: 重新开始", True, BLACK), (panel_x, panel_y))
        panel_y += 20
        self.screen.blit(self.small_font.render("S键: 设置", True, BLACK), (panel_x, panel_y))

    def handle_click(self, pos):
        """处理鼠标点击"""
        # [修改] 增加玩家回合判断
        if self.game_over or self.ai_thinking or \
           ("human" in self.game_mode and self.current_player != self.human_player):
            return

        x, y = pos

        # 检查是否点击在棋盘范围内
        if MARGIN <= x < WIDTH - MARGIN and MARGIN <= y < HEIGHT - MARGIN:
            col = round((x - MARGIN) / SQUARE_SIZE)
            row = round((y - MARGIN) / SQUARE_SIZE)

            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                action = row * BOARD_SIZE + col
                if action in self.env.get_valid_actions():
                    # [修改] 不再在这里触发AI，只执行人类的移动
                    self.make_move(action)

    def make_move(self, action):
        """执行一步棋，并只切换玩家和检查游戏结束"""
        # [简化] 这个函数现在非常纯粹
        self.env.step(action)
        self.zero_mcts_player.update_root(action)
        
        # 检查游戏结束
        if self.env._is_terminal():
            self.game_over = True
            self.winner = self.env.winner
            # 游戏结束，直接返回，不再切换玩家
            return
        
        # 切换玩家
        self.current_player = 3 - self.current_player

    def ai_move(self):
        """AI下棋"""
        if self.game_over:
            return

        self.ai_thinking = True

        try:
            if self.game_mode == "human_vs_ai":
                # 人类 vs AI 模式，使用ZeroMCTS
                if self.zero_mcts_player:
                    action = self.zero_mcts_player.run(self.zero_iterations, use_dirichlet=False)
                    self.zero_mcts_player.update_root(action)
                else:
                    # 回退到纯MCTS
                    action = self.pure_mcts_player.run(self.ai_iterations)

            elif self.game_mode == "ai_vs_ai":
                # AI vs AI 模式
                if self.current_player == 1:
                    # 黑棋使用ZeroMCTS
                    if self.zero_mcts_player:
                        action = self.zero_mcts_player.run(self.zero_iterations, use_dirichlet=False)
                        self.zero_mcts_player.update_root(action)
                    else:
                        action = self.pure_mcts_player.run(self.ai_iterations)
                else:
                    # 白棋使用纯MCTS
                    action = self.pure_mcts_player.run(self.ai_iterations)
                    self.zero_mcts_player.update_root(action)

            # 延迟一下让AI思考更真实
            time.sleep(0.5)

            # 在主线程中执行移动
            pygame.event.post(pygame.event.Event(pygame.USEREVENT, {'action': action}))

        except Exception as e:
            print(f"AI move error: {str(e)}")
        finally:
            self.ai_thinking = False

    def reset_game(self):
        """重置游戏"""
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game_over = False
        self.current_player = 1
        self.winner = None
        self.ai_thinking = False

        # 重置AI状态
        if self.zero_mcts_player:
            self.zero_mcts_player = ZeroMCTS(self.env.clone(), self.zero_policy, device=self.device)
        self.pure_mcts_player = MCTS(self.env.clone(), RandomStrategy(), c=1.41)

    def show_settings_dialog(self):
        """显示设置对话框"""
        dialog = tk.Tk()
        dialog.title("游戏设置")
        dialog.geometry("300x200")

        # AI迭代次数
        ttk.Label(dialog, text="AI迭代次数:").pack(pady=5)
        iterations_var = tk.IntVar(value=self.ai_iterations)
        iterations_scale = ttk.Scale(dialog, from_=100, to=2000, variable=iterations_var,
                                   orient=tk.HORIZONTAL, length=200)
        iterations_scale.pack()
        iterations_label = ttk.Label(dialog, text=str(self.ai_iterations))
        iterations_label.pack()

        def update_iterations(value):
            iterations_label.config(text=str(int(float(value))))

        iterations_scale.config(command=update_iterations)

        # AI温度
        ttk.Label(dialog, text="AI温度 (探索性):").pack(pady=5)
        temp_var = tk.DoubleVar(value=self.ai_temperature)
        temp_scale = ttk.Scale(dialog, from_=0.0, to=1.0, variable=temp_var,
                             orient=tk.HORIZONTAL, length=200)
        temp_scale.pack()
        temp_label = ttk.Label(dialog, text=str(self.ai_temperature))
        temp_label.pack()

        def update_temp(value):
            temp_label.config(text=f"{float(value):.2f}")

        temp_scale.config(command=update_temp)

        def apply_settings():
            self.ai_iterations = iterations_var.get()
            self.ai_temperature = temp_var.get()
            dialog.destroy()

        ttk.Button(dialog, text="应用", command=apply_settings).pack(pady=20)

        dialog.mainloop()

    def run(self):
        """主循环"""
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # 切换游戏模式
                        modes = ["human_vs_ai", "ai_vs_ai", "human_vs_human"]
                        current_idx = modes.index(self.game_mode)
                        self.game_mode = modes[(current_idx + 1) % len(modes)]
                        self.reset_game()
                    elif event.key == pygame.K_r:
                        # 重新开始
                        self.reset_game()
                    elif event.key == pygame.K_s:
                        # 设置
                        self.show_settings_dialog()
                elif event.type == pygame.USEREVENT:
                    if 'action' in event.dict:
                        self.make_move(event.dict['action'])

            if not self.game_over and not self.ai_thinking:
                is_human_turn = (self.game_mode == "human_vs_ai" and self.current_player == self.human_player) or \
                                (self.game_mode == "human_vs_human")
                
                # 如果当前不是人类的回合，那就是AI的回合
                if not is_human_turn:
                    # 启动AI思考线程
                    threading.Thread(target=self.ai_move, daemon=True).start()
            # 绘制
            self.draw_board()
            self.draw_pieces()
            if self.show_control_panel:
                self.draw_control_panel()

            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    game = GomokuBattleGUI()
    game.run()
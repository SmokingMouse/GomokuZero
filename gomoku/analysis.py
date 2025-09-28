# gui_optimized.py

import pygame
import threading
import time
import numpy as np
import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcts import ZeroMCTS

# --- 常量定义 (保持不变) ---
BOARD_SIZE = 9
SQUARE_SIZE = 40
MARGIN = 40
WIDTH = HEIGHT = SQUARE_SIZE * (BOARD_SIZE - 1) + MARGIN * 2
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BOARD_COLOR = (210, 180, 140)
PROB_COLOR_BASE = (255, 0, 0)

class GameGUI:
    def __init__(self, human_player_color='black'):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Gomoku - Human vs. AlphaZero AI")
        self.clock = pygame.time.Clock() # <--- [优化1] 增加时钟以控制帧率

        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.human_player = 1 if human_player_color == 'black' else 2
        
        # --- AI 设置 ---
        self.ai_policy = ZeroPolicy(board_size=BOARD_SIZE, num_blocks=2).to('cpu')
        # 请务必取消这行注释并加载你的模型!
        self.ai_policy.load_state_dict(torch.load('continue_model/policy_step_350000.pth', map_location='cpu'))
        self.ai_policy.eval()

        self.helper_policy = ZeroPolicy(board_size=BOARD_SIZE).to('cpu')
        self.helper_policy.load_state_dict(torch.load('continue_model/policy_step_660000.pth', map_location='cpu'))
        self.helper_policy.eval()
        
        # --- [优化2] 统一的 MCTS 实例 ---
        # AI 对手和后台搜索将共享这一个 MCTS 实例
        self.mcts = ZeroMCTS(self.ai_policy, device='cpu')
        self.analysis_mcts = ZeroMCTS(self.helper_policy, device='cpu')

        # self.helper = ZeroMCTS(self.ai_policy, device='cpu')

        # --- 后台搜索线程设置 ---
        self.search_thread = None
        self.searching = False
        self.prob_map_lock = threading.Lock() 
        self.prob_map = np.zeros((BOARD_SIZE, BOARD_SIZE))

        # --- [优化3] 预渲染 Surface ---
        self.prob_square_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)

        # --- [新] 添加胜率图和总模拟次数变量 ---
        self.win_rate_map = np.zeros((BOARD_SIZE, BOARD_SIZE))
        self.total_simulations = 0

        self.game_over = False
        self.font = pygame.font.Font(None, 42)
        self.small_font = pygame.font.Font(None, 18)

    def draw_board(self):
        # ... (此函数保持不变) ...
        self.screen.fill(BOARD_COLOR)
        for i in range(BOARD_SIZE):
            pygame.draw.line(self.screen, BLACK, (MARGIN, MARGIN + i * SQUARE_SIZE), (WIDTH - MARGIN, MARGIN + i * SQUARE_SIZE), 2)
            pygame.draw.line(self.screen, BLACK, (MARGIN + i * SQUARE_SIZE, MARGIN), (MARGIN + i * SQUARE_SIZE, HEIGHT - MARGIN), 2)


    def draw_pieces(self):
        # ... (此函数保持不变) ...
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.env.board[r, c] == 1:
                    pygame.draw.circle(self.screen, BLACK, (MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE), SQUARE_SIZE // 2 - 2)
                elif self.env.board[r, c] == 2:
                    pygame.draw.circle(self.screen, WHITE, (MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE), SQUARE_SIZE // 2 - 2)


    def draw_prob_heatmap(self):
        # --- [优化3] 使用预渲染的 Surface ---
        with self.prob_map_lock:
            current_prob_map = self.prob_map.copy()

        if np.sum(current_prob_map) == 0: return

        max_prob = np.max(current_prob_map)
        if max_prob > 0:
            normalized_map = current_prob_map / max_prob
        else:
            return

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.env.board[r, c] == 0 and normalized_map[r, c] > 0.01:
                    alpha = int(200 * normalized_map[r, c]) # 调整最大透明度
                    self.prob_square_surface.fill(PROB_COLOR_BASE + (alpha,))
                    self.screen.blit(self.prob_square_surface, (MARGIN + (c-0.5) * SQUARE_SIZE, MARGIN + (r-0.5) * SQUARE_SIZE))
    
    def display_message(self, message):
        with self.prob_map_lock:
            sims = self.total_simulations
        
        full_message = f"{message} (Sims: {sims})"

        text = self.font.render(full_message, True, BLACK, (230,220,200)) # slightly different bg for better reading
        text_rect = text.get_rect(center=(WIDTH // 2, MARGIN // 2))
        self.screen.blit(text, text_rect)

    def _background_search_loop(self):
        """后台持续运行 MCTS，并周期性更新概率图和胜率图"""
        print("Background search thread started.")
        last_update_time = 0
        while self.searching:
            self.analysis_mcts.run(self.env, iterations=200, use_dirichlet=False) 
            
            current_time = time.time()
            if current_time - last_update_time > 0.2:
                last_update_time = current_time
                
                # --- [修改] 提取更多数据 ---
                visits = np.zeros(self.env.board_size**2)
                q_values = np.zeros(self.env.board_size**2) # 用来存储每个动作的Q值
                
                if self.analysis_mcts.root:
                    for action, node in self.analysis_mcts.root.children.items():
                        visits[action] = node.visits
                        # 注意：q_value 是从当前玩家视角的价值。
                        # MCTS内部的q_value是(-1, 1)范围，我们将其转换为(0, 1)的胜率。
                        q_values[action] = (-node.q_value + 1) / 2.0
                
                total_visits = np.sum(visits)
                
                if total_visits > 0:
                    probs = visits / total_visits
                    # --- [修改] 加锁并更新所有共享数据 ---
                    with self.prob_map_lock:
                        self.prob_map = probs.reshape(self.env.board_size, self.env.board_size)
                        self.win_rate_map = q_values.reshape(self.env.board_size, self.env.board_size)
                        # self.mcts.root.visits 包含了根节点被访问的总次数，即总模拟次数
                        self.total_simulations = self.analysis_mcts.root.visits 
                
        print("Background search thread stopped.")

    def draw_win_rates(self):
        """在棋盘上绘制每个候选点的胜率"""
        with self.prob_map_lock:
            # 创建一个副本以避免在渲染时数据被修改
            current_win_rate_map = self.win_rate_map.copy()

        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                # 只显示有意义的胜率 (例如，概率大于1%的点)
                if self.prob_map[r, c] > 0.01:
                    win_rate = current_win_rate_map[r, c]
                    
                    # 根据胜率决定文本颜色，提高可读性
                    # 胜率高 -> 白色文本 (因为热力图背景是红色)
                    # 胜率低 -> 黑色文本
                    text_color = WHITE if win_rate > 0.6 else BLACK
                    
                    # 格式化文本为百分比
                    text_surface = self.small_font.render(f"{win_rate:.0%}", True, text_color)
                    
                    # 计算文本位置，使其居中于格子
                    text_rect = text_surface.get_rect(center=(MARGIN + c * SQUARE_SIZE, MARGIN + r * SQUARE_SIZE))
                    
                    self.screen.blit(text_surface, text_rect)

    def start_background_search(self):
        if not self.searching:
            self.searching = True
            # [优化2] 后台线程现在直接操作共享的 self.mcts 实例
            self.search_thread = threading.Thread(target=self._background_search_loop, daemon=True)
            self.search_thread.start()

    def stop_background_search(self):
        if self.searching:
            self.searching = False
            if self.search_thread:
                self.search_thread.join(timeout=0.5) # 等待线程结束
            with self.prob_map_lock:
                self.prob_map.fill(0)
                self.win_rate_map.fill(1)
                self.total_simulations = 0

    def handle_human_move(self, pos):
        if self.game_over or self.env.current_player != self.human_player:
            return

        col = round((pos[0] - MARGIN) / SQUARE_SIZE)
        row = round((pos[1] - MARGIN) / SQUARE_SIZE)
        
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
            action = row * BOARD_SIZE + col
            if action in self.env.get_valid_actions():
                self.stop_background_search()

                self.env.step(action)
                # [优化2] AI 直接继承人类思考的结果，只需推进树即可
                # self.mcts.update_root(action) 
                
                self.check_game_over()
                # AI 的回合将在主循环中被触发，而不是在这里直接调用

    def ai_move(self):
        """这是一个非阻塞的AI思考启动器"""
        if self.game_over or self.env.current_player == self.human_player:
            return
        
        # AI 使用固定次数进行搜索，这里我们直接从 MCTS 实例中获取最优动作
        # 因为后台线程可能已经为AI的这一步做了很多搜索
        # ai_action = self.mcts.select_action_with_temperature(temperature=0)[0]
        ai_action, _ = self.mcts.run(self.env, 200, False)
        
        self.env.step(ai_action)
        # self.mcts.update_root(ai_action)
        self.check_game_over()
        
        # 切换回人类回合，开始后台搜索
        if not self.game_over:
            self.start_background_search()

    def check_game_over(self):
        if self.env._is_terminal():
            self.game_over = True
            self.stop_background_search()

    def run(self):
        running = True
        
        if self.human_player == 1:
            self.start_background_search()

        while running:
            # --- 事件处理 ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_human_move(pygame.mouse.get_pos())

            # --- 游戏逻辑更新 ---
            if not self.game_over and self.env.current_player != self.human_player:
                # [优化4] 将AI的移动也放在主循环中，使其非阻塞
                self.ai_move()

            # --- 渲染 ---
            self.draw_board()
            self.draw_prob_heatmap()
            self.draw_pieces()
            self.draw_win_rates() 
            
            if self.game_over:
                winner_map = {1: "Black", 2: "White", 0: "Draw"}
                self.display_message(f"Game Over! Winner: {winner_map.get(self.env.winner, 'Unknown')}")
            elif self.env.current_player == self.human_player:
                self.display_message("Your Turn")
            else:
                self.display_message("AI's Turn")

            pygame.display.flip()
            
            # --- [优化1] 控制帧率 ---
            self.clock.tick(30) # 限制主循环最高为 30 FPS

        self.stop_background_search()
        pygame.quit()

if __name__ == "__main__":
    # 你可以选择执黑或执白
    human_color = 'black' 
    # human_color = 'white' 

    game = GameGUI(human_player_color=human_color)
    game.run()
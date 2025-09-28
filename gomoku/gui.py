import pygame
import tkinter as tk
from tkinter import ttk
import torch
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
from gomoku.zero_mcts import ZeroMCTS
from gomoku.mcts import MCTS

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


# 颜色配置
BACKGROUND = (220, 179, 92)
BTN_FILL = (245, 245, 245)
BTN_ACTIVE = (255, 255, 0)
BTN_BORDER = (60, 60, 60)

class ZeroPlayer():
    def __init__(self, iter = 200, use_dirichlet = False):
        zero_policy = ZeroPolicy(board_size=BOARD_SIZE, num_blocks=2)
        zero_policy.load_state_dict(torch.load('continue_model/policy_step_350000.pth', map_location=torch.device('cpu')))
        zero_policy.eval()
        self.zero_mcts_player = ZeroMCTS(zero_policy)
        self.iter = iter
        self.use_dirichlet = use_dirichlet
    
    def run(self, env):
        action, _ = self.zero_mcts_player.run(env, self.iter, self.use_dirichlet)
        return action

class MctsPlayer():
    def __init__(self, iter):
        self.mcts = MCTS()
        self.iter = iter 
    
    def run(self, env):
        return self.mcts.run(env, self.iter)

class GomokuBattleGUI:
    def __init__(self):
        pygame.init()
        
        # 定义布局常量
        self.CONTROL_PANEL_WIDTH = 300
        self.BUTTON_HEIGHT = 30
        self.BUTTON_WIDTH = 80
        self.BUTTON_SPACING = 10
        self.SLIDER_HEIGHT = 60
        self.CONTROL_AREA_HEIGHT = 200
        
        # 计算总窗口大小
        self.WINDOW_WIDTH = WIDTH + self.CONTROL_PANEL_WIDTH
        self.WINDOW_HEIGHT = max(HEIGHT, 600)  # 确保足够高度显示控制面板
        
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        pygame.display.set_caption("Gomoku AI Battle Arena")
        self.clock = pygame.time.Clock()

        try:
            # 定义一个包含多个中文字体名称的列表，按优先级排序
            # 'WenQuanYi Zen Hei' 是在 Ubuntu/WSL 中安装的字体
            # 'Microsoft YaHei UI' 等作为在 Windows 原生环境运行时的备选
            font_names = "STHeiti Light, Hiragino Sans GB, WenQuanYi Zen Hei, Microsoft YaHei UI, SimHei, SimSun"
            
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
        self.game_over = False
        self.current_player = 1  # 1 for black, 2 for white
        self.winner = None

        # AI设置
        self.zero_iterations = 400
        self.ai_iterations = 40000
        self.pure_mcts_iterations = 1000  # 纯MCTS迭代次数配置
        self.device = 'cpu'

        # AI状态管理
        self.ai_thinking = False
        self.last_ai_move_time = 0
        self.ai_move_delay = 500  # AI移动延迟（毫秒），减少延迟让响应更快
        
        # AI辅助功能
        self.show_ai_assistance = False  # 是否显示AI辅助
        self.ai_policy_values = None  # AI策略值
        self.ai_win_rate = None  # AI胜率评估
        
        # 控制面板
        self.show_control_panel = True

        # 玩家类型定义 (简化版)
        self.PLAYER_TYPES = {
            'human': '人类',
            'zero_mcts': 'Zero MCTS', 
            'pure_mcts': '纯MCTS'
        }
        
        # 黑棋和白棋的玩家类型
        self.black_player_type = 'zero_mcts'  # 默认黑棋使用Zero MCTS
        self.white_player_type = 'human'      # 默认白棋是人类

        self.p1_player = None
        self.p2_player = None

        self.update_player(1, self.black_player_type)
        self.update_player(2, self.white_player_type)
        
        # 按钮布局
        self.setup_layout()

    def setup_layout(self):
        """设置鲁棒的布局系统"""
        # 定义布局区域
        self.board_area = pygame.Rect(0, 0, WIDTH, HEIGHT)
        self.control_panel_area = pygame.Rect(WIDTH, 0, self.CONTROL_PANEL_WIDTH, self.WINDOW_HEIGHT)
        
        # 控制面板内部的布局
        margin = 20
        
        # 标题区域 (顶部)
        self.title_y = margin
        
        # 游戏信息区域
        self.info_y = self.title_y + 60
        
        # 按钮区域 (简化布局)
        button_rows = 3  # 增加一行用于配置按钮
        total_button_height = button_rows * self.BUTTON_HEIGHT + (button_rows - 1) * self.BUTTON_SPACING
        start_y = self.WINDOW_HEIGHT - total_button_height - margin - 20
        
        self.buttons = {}
        button_names = [
            ['black_human', 'black_zero_mcts', 'black_pure_mcts'],
            ['white_human', 'white_zero_mcts', 'white_pure_mcts'],
            ['config', None, None]  # 配置按钮
        ]
        
        for row_idx, row in enumerate(button_names):
            for col_idx, name in enumerate(row):
                if name is None:
                    continue
                    
                x = WIDTH + margin + col_idx * (self.BUTTON_WIDTH + self.BUTTON_SPACING)
                y = start_y + row_idx * (self.BUTTON_HEIGHT + self.BUTTON_SPACING)
                
                self.buttons[name] = pygame.Rect(x, y, self.BUTTON_WIDTH, self.BUTTON_HEIGHT)

    def setup_button_layout(self):
        """设置按钮布局 - 保持向后兼容"""
        self.setup_layout()

    def draw_buttons(self):
        """绘制控制按钮 - 玩家类型选择版本"""
        for name, rect in self.buttons.items():
            # 边界检查：确保按钮在有效区域内
            if (rect.x < 0 or rect.y < 0 or 
                rect.x + rect.width > self.WINDOW_WIDTH or 
                rect.y + rect.height > self.WINDOW_HEIGHT):
                continue
                
            # 按钮背景 - 根据当前选择的玩家类型高亮
            color = BTN_FILL  # 默认颜色
            
            if name.startswith('black_'):
                player_type = name.replace('black_', '')
                if self.black_player_type == player_type:
                    color = BTN_ACTIVE
            elif name.startswith('white_'):
                player_type = name.replace('white_', '')
                if self.white_player_type == player_type:
                    color = BTN_ACTIVE
            elif name in ['play', 'back', 'forward']:
                color = BTN_FILL
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BTN_BORDER, rect, 2)
            
            # 按钮文字
            text_map = {
                'black_human': '黑:人类',
                'black_zero_mcts': '黑:Zero',
                'black_pure_mcts': '黑:纯MCTS',
                'white_human': '白:人类',
                'white_zero_mcts': '白:Zero', 
                'white_pure_mcts': '白:纯MCTS',
                'config': '配置'
            }
            text = self.small_font.render(text_map[name], True, BLACK)
            
            # 确保文字在按钮内居中
            tx = rect.x + max(1, (rect.width - text.get_width()) // 2)
            ty = rect.y + max(1, (rect.height - text.get_height()) // 2)
            self.screen.blit(text, (tx, ty))

    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(BACKGROUND)  # 使用增强的背景色

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
        """绘制右侧控制面板 - 使用相对布局"""
        # 绘制面板背景
        pygame.draw.rect(self.screen, (240, 240, 240), self.control_panel_area)
        pygame.draw.line(self.screen, BLACK, (WIDTH, 0), (WIDTH, self.WINDOW_HEIGHT), 2)

        # 使用相对布局绘制各个组件
        margin = 20
        x = WIDTH + margin
        y = margin

        # 标题
        title = self.font.render("控制面板", True, BLACK)
        self.screen.blit(title, (x, y))
        y += 40

        # 玩家类型显示
        black_type = self.PLAYER_TYPES.get(self.black_player_type, '未知')
        white_type = self.PLAYER_TYPES.get(self.white_player_type, '未知')
        
        black_text = self.small_font.render(f"黑棋: {black_type}", True, BLACK)
        self.screen.blit(black_text, (x, y))
        y += 20
        
        white_text = self.small_font.render(f"白棋: {white_type}", True, BLACK)
        self.screen.blit(white_text, (x, y))
        y += 25

        # 纯MCTS配置显示
        mcts_text = self.small_font.render(f"纯MCTS搜索: {self.pure_mcts_iterations}次", True, BLACK)
        self.screen.blit(mcts_text, (x, y))
        y += 20

        # 当前玩家
        player_text = self.small_font.render(f"当前: {'黑棋' if self.current_player == 1 else '白棋'}", True, BLACK)
        self.screen.blit(player_text, (x, y))
        y += 25

        # 游戏状态
        if self.game_over:
            if self.winner == 0 or self.winner is None:
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
        self.screen.blit(status, (x, y))
        y += 35

        # 操作说明
        self.screen.blit(self.small_font.render("操作说明:", True, BLACK), (x, y))
        y += 20
        
        instructions = [
            "点击棋盘: 下棋（人类回合）",
            "选择玩家类型: 点击黑/白按钮",
            "C键: MCTS配置",
            "R键: 重新开始"
        ]
        
        for instruction in instructions:
            self.screen.blit(self.small_font.render(instruction, True, BLACK), (x, y))
            y += 18
        
        # 绘制按钮
        self.draw_buttons()

    def handle_click(self, pos):
        """处理鼠标点击 - 简化版本"""
        x, y = pos
        
        # 检查按钮点击
        for name, rect in self.buttons.items():
            if rect.collidepoint(pos):
                self.handle_button_click(name)
                return
        
        # 检查棋盘点击 - 简化逻辑
        if self.game_over:
            return
            
        # 只有当前玩家是人类时才能点击棋盘
        current_player_type = self.black_player_type if self.current_player == 1 else self.white_player_type
        if current_player_type != 'human':
            return

        # 检查是否点击在棋盘范围内
        if MARGIN <= x < WIDTH - MARGIN and MARGIN <= y < HEIGHT - MARGIN:
            col = round((x - MARGIN) / SQUARE_SIZE)
            row = round((y - MARGIN) / SQUARE_SIZE)

            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                action = row * BOARD_SIZE + col
                if action in self.env.get_valid_actions():
                    self.make_move(action)

    def update_player(self, player: int, player_type):
        new_player = None
        if player_type == 'zero_mcts':
            new_player = ZeroPlayer(self.zero_iterations)
        elif player_type == 'pure_mcts':
            new_player = MctsPlayer(self.pure_mcts_iterations)
        
        if player == 1:
            self.p1_player = new_player
        else:
            self.p2_player = new_player

    def handle_button_click(self, button_name):
        """处理按钮点击 - 简化的玩家类型选择"""
        if button_name.startswith('black_'):
            # 设置黑棋玩家类型
            player_type = button_name.replace('black_', '')
            self.black_player_type = player_type
            self.update_player(1, self.black_player_type)
            
        elif button_name.startswith('white_'):
            # 设置白棋玩家类型
            player_type = button_name.replace('white_', '')
            self.white_player_type = player_type
            self.update_player(2, self.white_player_type)
            
        elif button_name == 'config':
            # 打开配置对话框
            self.show_mcts_config_dialog()
            
    def make_move(self, action):
        """执行一步棋 - 简化版本"""
        # 执行移动
        self.env.step(action)
        
        # 检查游戏结束
        if self.env._is_terminal():
            self.game_over = True
            self.winner = self.env.winner
            return
        
        # 切换玩家
        self.current_player = 3 - self.current_player

    def get_player_move(self, player):
        if player == 1:
            return self.p1_player.run(self.env)
        else:
            return self.p2_player.run(self.env)

    def move(self):
        if self.game_over:
            return
        
        # 根据玩家类型获取移动
        action = self.get_player_move(self.current_player)
        
        if action is not None:
            self.make_move(action)

    def reset_game(self):
        """重置游戏"""
        self.env = GomokuEnv(board_size=BOARD_SIZE)
        self.game_over = False
        self.current_player = 1
        self.winner = None
        self.ai_thinking = False
        
        # 重置AI状态
        self.update_player(1, self.black_player_type)
        self.update_player(2, self.white_player_type)

    # def show_player_selection_dialog(self):
    #     """显示玩家选择对话框"""
    #     dialog = tk.Tk()
    #     dialog.title("玩家设置")
    #     dialog.geometry("300x250")

    #     # 人类玩家选择
    #     ttk.Label(dialog, text="人类玩家:").pack(pady=5)
    #     player_var = tk.IntVar(value=self.human_player)
    #     ttk.Radiobutton(dialog, text="黑棋 (先手)", variable=player_var, value=1).pack()
    #     ttk.Radiobutton(dialog, text="白棋 (后手)", variable=player_var, value=2).pack()

    #     # 是否先手
    #     ttk.Label(dialog, text="是否先手:").pack(pady=5)
    #     first_var = tk.BooleanVar(value=self.human_first)
    #     ttk.Radiobutton(dialog, text="先手", variable=first_var, value=True).pack()
    #     ttk.Radiobutton(dialog, text="后手", variable=first_var, value=False).pack()

    #     def apply_settings():
    #         self.human_player = player_var.get()
    #         self.human_first = first_var.get()
    #         # 重新设置游戏以应用更改
    #         self.reset_game()
    #         dialog.destroy()

    #     ttk.Button(dialog, text="应用", command=apply_settings).pack(pady=20)
    #     dialog.mainloop()

    # def show_settings_dialog(self):
    #     """显示设置对话框"""
    #     dialog = tk.Tk()
    #     dialog.title("游戏设置")
    #     dialog.geometry("300x200")

    #     # AI迭代次数
    #     ttk.Label(dialog, text="AI迭代次数:").pack(pady=5)
    #     iterations_var = tk.IntVar(value=self.ai_iterations)
    #     iterations_scale = ttk.Scale(dialog, from_=100, to=2000, variable=iterations_var,
    #                                orient=tk.HORIZONTAL, length=200)
    #     iterations_scale.pack()
    #     iterations_label = ttk.Label(dialog, text=str(self.ai_iterations))
    #     iterations_label.pack()

    #     def update_iterations(value):
    #         iterations_label.config(text=str(int(float(value))))

    #     iterations_scale.config(command=update_iterations)

    #     # AI温度
    #     ttk.Label(dialog, text="AI温度 (探索性):").pack(pady=5)
    #     temp_var = tk.DoubleVar(value=self.ai_temperature)
    #     temp_scale = ttk.Scale(dialog, from_=0.0, to=1.0, variable=temp_var,
    #                          orient=tk.HORIZONTAL, length=200)
    #     temp_scale.pack()
    #     temp_label = ttk.Label(dialog, text=str(self.ai_temperature))
    #     temp_label.pack()

    #     def update_temp(value):
    #         temp_label.config(text=f"{float(value):.2f}")

    #     temp_scale.config(command=update_temp)

    #     def apply_settings():
    #         self.ai_iterations = iterations_var.get()
    #         self.ai_temperature = temp_var.get()
    #         dialog.destroy()

    #     ttk.Button(dialog, text="应用", command=apply_settings).pack(pady=20)

    #     dialog.mainloop()

    def show_mcts_config_dialog(self):
        """显示纯MCTS配置对话框 - macOS兼容版本"""
        try:
            # 暂停pygame事件循环
            pygame.event.set_blocked(None)
            
            dialog = tk.Toplevel()
            dialog.title("MCTS配置")
            dialog.geometry("300x200")
            dialog.resizable(False, False)
            
            # 当前配置显示
            ttk.Label(dialog, text=f"当前纯MCTS搜索次数: {self.pure_mcts_iterations}").pack(pady=10)
            
            # 滑块控制
            ttk.Label(dialog, text="搜索次数:").pack(pady=5)
            iterations_var = tk.IntVar(value=self.pure_mcts_iterations)
            iterations_scale = ttk.Scale(dialog, from_=100, to=10000, variable=iterations_var,
                                       orient=tk.HORIZONTAL, length=250)
            iterations_scale.pack()
            
            # 数值显示
            iterations_label = ttk.Label(dialog, text=str(self.pure_mcts_iterations))
            iterations_label.pack()
            
            def update_iterations(value):
                iterations_label.config(text=str(int(float(value))))
            
            iterations_scale.config(command=update_iterations)
            
            # 预设按钮
            preset_frame = ttk.Frame(dialog)
            preset_frame.pack(pady=10)
            
            def set_preset(value):
                iterations_var.set(value)
                iterations_label.config(text=str(value))
                iterations_scale.set(value)
            
            ttk.Button(preset_frame, text="快速(500)", command=lambda: set_preset(500)).pack(side=tk.LEFT, padx=5)
            ttk.Button(preset_frame, text="标准(2000)", command=lambda: set_preset(2000)).pack(side=tk.LEFT, padx=5)
            ttk.Button(preset_frame, text="深度(5000)", command=lambda: set_preset(5000)).pack(side=tk.LEFT, padx=5)
            
            def apply_config():
                self.pure_mcts_iterations = iterations_var.get()
                # 更新当前纯MCTS玩家
                if self.black_player_type == 'pure_mcts':
                    self.update_player(1, 'pure_mcts')
                if self.white_player_type == 'pure_mcts':
                    self.update_player(2, 'pure_mcts')
                dialog.destroy()
            
            def cancel_config():
                dialog.destroy()
            
            # 按钮区域
            button_frame = ttk.Frame(dialog)
            button_frame.pack(pady=20)
            
            ttk.Button(button_frame, text="应用", command=apply_config).pack(side=tk.LEFT, padx=10)
            ttk.Button(button_frame, text="取消", command=cancel_config).pack(side=tk.LEFT, padx=10)
            
            # 设置对话框为模态
            dialog.transient()
            dialog.grab_set()
            
            # 等待对话框关闭
            self.screen.fill(BACKGROUND)  # 清空屏幕避免冲突
            pygame.display.flip()
            dialog.wait_window()
            
        except Exception as e:
            print(f"配置对话框错误: {e}")
            # 回退到简单的输入框
            self.simple_mcts_config()
        finally:
            # 恢复pygame事件循环
            pygame.event.set_allowed(None)

    def simple_mcts_config(self):
        """简单的MCTS配置回退方案"""
        # 使用pygame的文本输入方式（简化版）
        current_iterations = self.pure_mcts_iterations
        
        # 这里可以添加一个基于pygame的简单输入界面
        # 暂时使用简单的数值调整
        if current_iterations < 2000:
            self.pure_mcts_iterations = 2000
        elif current_iterations < 5000:
            self.pure_mcts_iterations = 5000
        else:
            self.pure_mcts_iterations = 1000
        
        print(f"MCTS搜索次数已调整为: {self.pure_mcts_iterations}")
        
        # 更新当前纯MCTS玩家
        if self.black_player_type == 'pure_mcts':
            self.update_player(1, 'pure_mcts')
        if self.white_player_type == 'pure_mcts':
            self.update_player(2, 'pure_mcts')

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
                    if event.key == pygame.K_c:
                        # MCTS配置
                        self.show_mcts_config_dialog()
                    elif event.key == pygame.K_r:
                        # 重新开始
                        self.reset_game()
                        continue

            # 优化的AI逻辑：带时序控制（更流畅）
            if not self.game_over and not self.ai_thinking:
                current_player_type = self.black_player_type if self.current_player == 1 else self.white_player_type
                if current_player_type != 'human':
                    current_time = pygame.time.get_ticks()
                    if current_time - self.last_ai_move_time >= self.ai_move_delay:
                        self.ai_thinking = True
                        # 使用较小的延迟，避免阻塞主循环
                        pygame.time.delay(50)  # 50ms延迟让UI更响应
                        self.move()
                        self.last_ai_move_time = current_time
                        self.ai_thinking = False
            
            # === 绘制 ===
            self.draw_board()
            self.draw_pieces()
            
            if self.show_control_panel:
                self.draw_control_panel()

            pygame.display.flip()
            self.clock.tick(60)  # 60FPS让界面更流畅

        pygame.quit()

if __name__ == "__main__":
    game = GomokuBattleGUI()
    game.run()
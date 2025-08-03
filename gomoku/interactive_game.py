import tkinter as tk
from tkinter import messagebox
from gomoku.gomoku_env import GomokuEnv
from gomoku.mcts import MCTS


class GomokuGUI:
    def __init__(self, board_size=15, cell_size=30):
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = 40
        
        # Initialize game
        self.env = GomokuEnv(board_size)
        self.env.reset()
        
        # Initialize AI
        self.ai = MCTS(strategy="random", c=1.41)
        self.ai_thinking = False
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("五子棋 - 人机对战")
        
        # Calculate window size
        window_size = board_size * cell_size + 2 * self.margin
        self.root.geometry(f"{window_size + 200}x{window_size}")
        
        # Create control panel
        self.create_control_panel()
        
        # Create canvas for board
        self.canvas = tk.Canvas(
            self.root,
            width=window_size,
            height=window_size,
            bg="#DEB887"
        )
        self.canvas.pack(side=tk.LEFT)
        
        # Bind click event
        self.canvas.bind("<Button-1>", self.on_click)
        
        # Draw board
        self.draw_board()
        
        # Game state
        self.game_over = False
        
    def create_control_panel(self):
        """Create control panel with buttons and info"""
        control_frame = tk.Frame(self.root, width=200, padx=10, pady=10)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Title
        title = tk.Label(control_frame, text="五子棋", font=("Arial", 16, "bold"))
        title.pack(pady=10)
        
        # Info labels
        self.info_frame = tk.Frame(control_frame)
        self.info_frame.pack(pady=10)
        
        self.current_player_label = tk.Label(
            self.info_frame, 
            text="当前玩家: 你 (X)", 
            font=("Arial", 12)
        )
        self.current_player_label.pack()
        
        self.status_label = tk.Label(
            self.info_frame, 
            text="游戏进行中...", 
            font=("Arial", 10)
        )
        self.status_label.pack()
        
        # Control buttons
        button_frame = tk.Frame(control_frame)
        button_frame.pack(pady=20)
        
        self.new_game_button = tk.Button(
            button_frame, 
            text="新游戏", 
            command=self.new_game,
            width=15,
            height=2
        )
        self.new_game_button.pack(pady=5)
        
        self.undo_button = tk.Button(
            button_frame,
            text="悔棋",
            command=self.undo_move,
            width=15,
            height=2
        )
        self.undo_button.pack(pady=5)
        
        # AI settings
        ai_frame = tk.LabelFrame(control_frame, text="AI设置", padx=5, pady=5)
        ai_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(ai_frame, text="思考深度:").pack()
        self.depth_var = tk.StringVar(value="1000")
        depth_entry = tk.Entry(ai_frame, textvariable=self.depth_var, width=10)
        depth_entry.pack()
        
        # Legend
        legend_frame = tk.LabelFrame(control_frame, text="图例", padx=5, pady=5)
        legend_frame.pack(pady=10, fill=tk.X)
        
        tk.Label(legend_frame, text="● 你 (X)").pack()
        tk.Label(legend_frame, text="○ AI (O)").pack()
        
    def draw_board(self):
        """Draw the game board"""
        self.canvas.delete("all")
        
        # Draw grid lines
        for i in range(self.board_size):
            # Vertical lines
            x = self.margin + i * self.cell_size
            self.canvas.create_line(
                x, self.margin,
                x, self.margin + (self.board_size - 1) * self.cell_size,
                width=1
            )
            
            # Horizontal lines
            y = self.margin + i * self.cell_size
            self.canvas.create_line(
                self.margin, y,
                self.margin + (self.board_size - 1) * self.cell_size, y,
                width=1
            )
            
            # Add coordinate labels
            if i % 2 == 0:  # Only label every other line to reduce clutter
                self.canvas.create_text(
                    x, self.margin - 10,
                    text=str(i),
                    font=("Arial", 8)
                )
                self.canvas.create_text(
                    self.margin - 10, y,
                    text=str(i),
                    font=("Arial", 8)
                )
        
        # Draw star points (for traditional Go board)
        star_points = [3, 7, 11] if self.board_size >= 15 else [2, 4, 6]
        for i in star_points:
            for j in star_points:
                if i < self.board_size and j < self.board_size:
                    x = self.margin + j * self.cell_size
                    y = self.margin + i * self.cell_size
                    self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="black")
    
    def draw_stones(self):
        """Draw all stones on the board"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                stone = self.env.board[row, col]
                if stone != 0:
                    self.draw_stone(row, col, stone)
    
    def draw_stone(self, row, col, player):
        """Draw a single stone"""
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2
        
        color = "black" if player == 1 else "white"
        outline = "black"
        
        self.canvas.create_oval(
            x - radius, y - radius,
            x + radius, y + radius,
            fill=color,
            outline=outline,
            width=2
        )
    
    def on_click(self, event):
        """Handle mouse click on board"""
        if self.game_over or self.ai_thinking:
            return
        
        # Calculate board position
        col = round((event.x - self.margin) / self.cell_size)
        row = round((event.y - self.margin) / self.cell_size)
        
        # Check if click is within board bounds
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            action = self.env._position_to_action(row, col)
            if action in self.env.get_valid_actions():
                self.make_move(action)
                
                if not self.game_over:
                    # AI's turn
                    self.root.after(100, self.ai_move)
    
    def make_move(self, action):
        """Make a move and update the display"""
        _, reward, done, _, _ = self.env.step(action)
        self.draw_stones()
        
        if done:
            self.game_over = True
            if reward > 0:
                winner = "你赢了！"
            elif reward < 0:
                winner = "AI赢了！"
            else:
                winner = "平局！"
            
            self.status_label.config(text=winner)
            messagebox.showinfo("游戏结束", winner)
        else:
            current_player = "AI" if self.env.current_player == 2 else "你"
            self.current_player_label.config(text=f"当前玩家: {current_player}")
    
    def ai_move(self):
        """AI makes a move using MCTS"""
        if self.game_over:
            return
            
        self.ai_thinking = True
        self.status_label.config(text="AI思考中...")
        
        try:
            iterations = int(self.depth_var.get())
            action = self.ai.run(self.env, iterations)
            self.make_ai_move(action)
        except Exception as error:
            self.handle_ai_error(str(error))
    
    def make_ai_move(self, action):
        """Execute AI move"""
        self.ai_thinking = False
        self.make_move(action)
    
    def handle_ai_error(self, error):
        """Handle AI errors"""
        self.ai_thinking = False
        self.status_label.config(text="AI错误")
        messagebox.showerror("AI错误", f"AI计算出错: {error}")
    
    def new_game(self):
        """Start a new game"""
        self.env.reset()
        self.game_over = False
        self.ai_thinking = False
        self.draw_board()
        self.draw_stones()
        self.status_label.config(text="游戏进行中...")
        self.current_player_label.config(text="当前玩家: 你 (X)")
    
    def undo_move(self):
        """Undo the last move"""
        # Note: This is a simplified undo - we reset and replay moves
        if len(self.env.move_history) >= 2:
            # Save current history
            history = self.env.move_history[:-2]
            
            # Reset game
            self.env.reset()
            self.game_over = False
            self.ai_thinking = False
            
            # Replay moves
            for row, col in history:
                action = self.env._position_to_action(row, col)
                self.env.step(action)
            
            self.draw_board()
            self.draw_stones()
            
            current_player = "AI" if self.env.current_player == 2 else "你"
            self.current_player_label.config(text=f"当前玩家: {current_player}")
            self.status_label.config(text="游戏进行中...")
    
    def run(self):
        """Start the game"""
        self.new_game()
        self.root.mainloop()


if __name__ == "__main__":
    game = GomokuGUI()
    game.run()
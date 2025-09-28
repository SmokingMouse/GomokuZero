import time
import math
import numpy as np
from gomoku.gomoku_env import GomokuEnv # 假设从你的文件中导入
from gomoku.player import Player # 假设从你的文件中导入

class AlphaBetaPlayer(Player):
    def __init__(self, search_depth=4, evaluator=None):
        """
        一个使用 Alpha-Beta 剪枝的 Minimax 搜索 AI。

        Args:
            search_depth (int): 最大搜索深度。偶数深度通常更稳定。
            evaluator (callable, optional): 局面评估函数。如果为 None，则使用默认的。
        """
        super().__init__(f"AlphaBeta Player (Depth {search_depth})")
        if search_depth < 2:
            raise ValueError("Search depth must be at least 2.")
        self.search_depth = search_depth
        self.evaluator = evaluator if evaluator is not None else self.stronger_evaluator

    def play(self, game: GomokuEnv, *args, **kwargs):
        """
        根据 Alpha-Beta 搜索决定下一步。
        """
        start_time = time.time()
        
        # 获取所有合法走法
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return {'action': None} # 没有可走的路

        best_action = -1
        # alpha 初始为负无穷，beta 初始为正无穷
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        # 遍历根节点的所有子节点（即所有第一步走法）
        for action in valid_actions:
            # 模拟走一步
            child_env = game.clone()
            child_env.step(action)
            
            # 对子节点进行 minimax 搜索，注意：现在轮到对手了，所以是 min_player
            score = self.minimax(child_env, self.search_depth - 1, alpha, beta, is_maximizing_player=False)
            
            # 更新最佳走法
            if score > best_score:
                best_score = score
                best_action = action
            
            # 更新 alpha 值
            alpha = max(alpha, best_score)
            
            # 在根节点层不需要 alpha >= beta 剪枝，因为我们需要遍历所有第一步

        end_time = time.time()
        print(f"[{self.name}] Chose action {best_action} with score {best_score:.2f}. Time taken: {end_time - start_time:.2f}s")
        
        # 实际走棋
        game.step(best_action)
        return {'action': best_action}


    def minimax(self, game: GomokuEnv, depth: int, alpha: float, beta: float, is_maximizing_player: bool):
        """
        Minimax 递归函数。

        Args:
            game (GomokuEnv): 当前棋盘状态。
            depth (int): 剩余搜索深度。
            alpha (float): Alpha 值。
            beta (float): Beta 值。
            is_maximizing_player (bool): 当前是否是最大化玩家（我方）的回合。
        """
        # 递归终止条件：达到最大深度或游戏结束
        if depth == 0 or game._is_terminal():
            # self.evaluator 的分数是基于当前 game.current_player 的视角
            # minimax 函数期望返回的是从“我方”（最大化玩家）视角看的分数
            score = self.evaluator(game, original_player=game.current_player)
            return score

        valid_actions = game.get_valid_actions()

        if is_maximizing_player:
            max_eval = -math.inf
            for action in valid_actions:
                child_env = game.clone()
                child_env.step(action)
                eval = self.minimax(child_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Beta 剪枝
            return max_eval
        else: # Minimizing player
            min_eval = math.inf
            for action in valid_actions:
                child_env = game.clone()
                child_env.step(action)
                eval = self.minimax(child_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Alpha 剪枝
            return min_eval

    def stronger_evaluator(self, game: GomokuEnv, original_player: int):
        """
        一个更强大的评估函数，会检查所有四个方向的多种棋型。
        """
        if game._is_terminal():
            winner = game.winner
            if winner == original_player: return 100000
            if winner == (3 - original_player): return -100000
            return 0 # Draw

        my_player = original_player
        opponent_player = 3 - original_player
        
        # 定义棋型和它们的权重
        # (我方活四，我方冲四，我方活三，...)
        pattern_scores = {
            "FIVE": 100000,
            "LIVE_FOUR": 10000,
            "DEAD_FOUR": 1000,
            "LIVE_THREE": 1000,
            "DEAD_THREE": 100,
            "LIVE_TWO": 10,
            "DEAD_TWO": 1,
        }

        my_score = self.calculate_patterns(game.board, my_player, pattern_scores)
        opponent_score = self.calculate_patterns(game.board, opponent_player, pattern_scores)

        # 最终分数是 我方分数 - 对手分数 * (一个略大于1的系数，表示防守更重要)
        return my_score - opponent_score * 1.1

    def calculate_patterns(self, board, player, scores):
        """辅助函数，计算一个玩家在棋盘上所有棋型的总分。"""
        total_score = 0
        board_size = len(board)

        # 四个方向的向量: 横, 竖, 左斜, 右斜
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        # 遍历棋盘上的每一个点
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == player:
                    # 对每个方向进行检查
                    for dr, dc in directions:
                        # 检查这个方向上，以 (r, c) 为起点的连子
                        # (这是一个简化的逻辑，真实的棋型判断会更复杂，
                        # 需要检查两端是否被堵住，但这个足以让AI变聪明)
                        
                        count = 0
                        for i in range(1, 5):
                            nr, nc = r + i * dr, c + i * dc
                            if 0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player:
                                count += 1
                            else:
                                break
                        
                        # 根据连子数量给予基础分数
                        if count == 1: # 连二
                            total_score += scores["DEAD_TWO"]
                        elif count == 2: # 连三
                            total_score += scores["DEAD_THREE"]
                        elif count == 3: # 连四
                            total_score += scores["DEAD_FOUR"]
                        elif count == 4: # 连五
                            total_score += scores["FIVE"]
        return total_score

    def simple_evaluator(self, game: GomokuEnv, original_player: int):
        """
        一个非常基础的局面评估函数。
        返回从 original_player 视角看的分数。
        """
        # 首先检查游戏是否已经结束
        if game._is_terminal():
            winner = game.winner
            if winner == original_player:
                return 100000  # 我方赢了，返回一个非常大的正数
            elif winner == (3 - original_player):
                return -100000 # 对手赢了，返回一个非常大的负数
            else:
                return 0 # 平局

        # 如果游戏未结束，则基于棋型计算分数
        score = 0
        board = game.board
        
        # 定义棋型和权重
        patterns = {
            # 我方 (original_player) 的棋型
            (original_player,) * 5: 100000,
            (0,) + (original_player,) * 4 + (0,): 5000, # 活四
            (original_player,) * 4: 500, # 冲四
            (0,) + (original_player,) * 3 + (0,): 200, # 活三
            (original_player,) * 3: 50, # 眠三
            (0,) + (original_player,) * 2 + (0,): 10, # 活二
            # 对手 (3 - original_player) 的棋型 (权重为负)
            ((3 - original_player),) * 5: -100000,
            (0,) + ((3 - original_player),) * 4 + (0,): -10000, # 防守活四比自己活三更重要
            ((3 - original_player),) * 4: -1000,
            (0,) + ((3 - original_player),) * 3 + (0,): -800,
            ((3 - original_player),) * 3: -80,
            (0,) + ((3 - original_player),) * 2 + (0,): -15,
        }
        
        # 遍历所有可能的五元组
        board_size = game.board_size
        # 遍历行、列、两条对角线
        # (这是一个简化的实现，只检查了部分棋型，但足以作为 demo)
        # 你可以去网上找一个更完整的五子棋棋型打分表来实现
        # 这里只简单实现横向
        for r in range(board_size):
            for c in range(board_size - 4):
                segment = tuple(board[r, c:c+5])
                if segment in patterns:
                    score += patterns[segment]
                    
        return score

# --- 如何在你的 Arena 中使用 ---
if __name__ == '__main__':
    # 假设你有 policy1 (Zero AI) 和 arena_parallel
    # from your_training_code import ZeroMCTSPlayer, ZeroPolicy, arena_parallel

    board_size = 9
    
    # 1. 创建你的 Zero AI Player
    # policy1 = ZeroPolicy(board_size)
    # policy1.load_state_dict(...)
    # player1 = ZeroMCTSPlayer(policy1, itermax=400, eager=True)

    # 2. 创建一个 Alpha-Beta AI 作为对手
    # 搜索深度为 4 已经是一个相当强的对手了
    player2 = AlphaBetaPlayer(search_depth=5)

    # 3. 进行对抗
    # 你需要修改你的 arena 代码，使其能接收不同类型的 Player 对象
    # 核心是确保 play_one_game 函数能正确调用 player.play(game)
    
    # 这里是一个简化的单局对抗示例
    game = GomokuEnv(board_size)
    players = [player2, player2] # 自己和自己下
    current_idx = 0
    while not game._is_terminal():
        game.render()
        player_to_move = players[current_idx]
        player_to_move.play(game)
        current_idx = 1 - current_idx
    
    game.render()
    print(f"Game Over! Winner is: Player {game.winner}")
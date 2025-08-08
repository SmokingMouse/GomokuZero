from tkinter import NO
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import gymnasium as gym
from gymnasium import spaces


class GomokuEnv(gym.Env):
    """
    Gomoku (Five in a Row) Environment
    
    The game is played on a 15x15 board where two players take turns
    placing stones. The first player to get 5 stones in a row (horizontally,
    vertically, or diagonally) wins.
    
    Action space: Discrete(225) - each action corresponds to a position on the board
    Observation space: Box(15, 15, 2) - the board state for both players
    """
    
    def __init__(self, board_size: int = 15, enable_history: bool = False):
        super().__init__()
        
        self.board_size = board_size
        self.action_space = spaces.Discrete(board_size * board_size)
        
        # Observation space: [player1_board, player2_board] as binary masks
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2, board_size, board_size), dtype=np.int8
        )
        
        # Game state
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = 1  # 1 or 2
        self.winner = None
        self.done = False

        # For tracking moves
        self.enable_history = enable_history
        self.move_history = []
        self.last_action = -1
        self.move_size = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1
        self.winner = None
        self.done = False
        self.move_history = []
        
        return self._get_observation(), self._get_info()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        if self.done:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Convert action to board coordinates
        row, col = self._action_to_position(action)
        
        # Check if move is valid
        if not self._is_valid_move(row, col):
            return self._get_observation(), -10.0, True, False, self._get_info()
        
        # Make the move
        self.board[row, col] = self.current_player
        self.move_size += 1
        self.move_history.append((row, col)) if self.enable_history else None
        self.last_action = action
        
        # Check for win
        if self._check_winner(row, col):
            self.winner = self.current_player
            self.done = True
            reward = 100.0 if self.current_player == 1 else -100.0
        elif self._is_board_full():
            self.done = True
            reward = 0.0  # Draw
        else:
            reward = 0.0
        self.current_player = 3 - self.current_player  # Switch player: 1 -> 2, 2 -> 1
        
        return self._get_observation(), reward, self.done, False, self._get_info()
    
    def _action_to_position(self, action: int) -> Tuple[int, int]:
        """Convert action index to board coordinates."""
        return action // self.board_size, action % self.board_size
    
    def _position_to_action(self, row: int, col: int) -> int:
        """Convert board coordinates to action index."""
        return row * self.board_size + col
    
    def _is_valid_move(self, row: int, col: int) -> bool:
        """Check if a move is valid."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size and self.board[row, col] == 0
    
    def _is_terminal(self) -> bool:
        """Check if the game is over (win or draw)."""
        return self.done or self.winner is not None
    
    def _check_winner(self, row: int, col: int) -> bool:
        """Check if the last move resulted in a win."""
        player = self.board[row, col]
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # horizontal, vertical, diagonal, anti-diagonal
        
        for dr, dc in directions:
            count = 1
            
            # Check in positive direction
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            # Check in negative direction
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False
    
    def _is_board_full(self) -> bool:
        """Check if the board is full (draw)."""
        return np.all(self.board != 0)
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation."""
        # Create binary masks for both players
        player1_board = (self.board == 1).astype(np.int8)
        player2_board = (self.board == 2).astype(np.int8)
        last_action_state = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        # last_action = self._action_to_position        
        if self.last_action != -1:
            last_action = self._action_to_position(self.last_action)
            last_action_state[last_action] = 1
        
        if self.current_player == 2:
            return np.stack([player2_board, player1_board, last_action_state], axis=0)

        return np.stack([player1_board, player2_board, last_action_state], axis=0)
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the current state."""
        return None
        # return {
        #     'current_player': self.current_player,
        #     'winner': self.winner,
        #     'move_history': self.move_history.copy(),
        #     'valid_actions': self.get_valid_actions()
        # }
    
    def get_valid_actions(self) -> List[int]:
        """Get list of valid actions (empty positions)."""
        valid_actions = []
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.board[row, col] == 0:
                    valid_actions.append(self._position_to_action(row, col))
        return valid_actions
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current board state as a string."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        board_str = "  " + " ".join([f"{i:2d}" for i in range(self.board_size)]) + "\n"
        
        for i, row in enumerate(self.board):
            board_str += f"{i:2d} " + " ".join([f"{symbols[cell]} " for cell in row]) + "\n"
        
        print(board_str)
        return None
    
    def get_board_string(self) -> str:
        """Get the board as a string for display purposes."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        result = []
        for row in self.board:
            result.append("".join(symbols[cell] for cell in row))
        return "\n".join(result)
    
    def clone(self) -> 'GomokuEnv':
        """Create a deep copy of the current environment state."""
        new_env = self.__class__.__new__(self.__class__)
        # new_env = GomokuEnv(self.board_size)
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.winner = self.winner
        new_env.done = self.done
        new_env.last_action = self.last_action
        new_env.move_size = self.move_size
        new_env.enable_history = self.enable_history
        new_env.board_size = self.board_size
        new_env.action_space = self.action_space
        new_env.observation_space = self.observation_space
        # new_env.move_history = self.move_history.copy()
        return new_env


class GomokuEnvSimple(GomokuEnv):
    """
    Simplified Gomoku environment with smaller board for faster training.
    """
    
    def __init__(self, board_size: int = 9):
        super().__init__(board_size=board_size)


# Example usage and testing
if __name__ == "__main__":
    env = GomokuEnv()
    obs, info = env.reset()
    
    print("Gomoku Environment initialized!")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial valid actions: {len(info['valid_actions'])}")
    
    # Test a few moves
    valid_actions = info['valid_actions']
    if valid_actions:
        action = valid_actions[0]
        obs, reward, done, truncated, info = env.step(action)
        print(f"Made move {action}, reward: {reward}")
        env.render()
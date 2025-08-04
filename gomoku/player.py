
# %% 
from gomoku.mcts import MCTS, ZeroMCTS
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
import random
import torch
# %%

class Player:
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return f"Player(name={self.name})"

    def __eq__(self, other):
        if isinstance(other, Player):
            return self.name == other.name
        return False
    
    def play(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class MCTSPlayer(Player):
    def __init__(self, game: GomokuEnv, itermax=2000):
        super().__init__("MCTS Player")
        self.game = game
        self.itermax = itermax

    def play(self):
        # Implement MCTS logic here
        mcts = MCTS(self.game) 
        action = mcts.run(self.itermax)

        self.game.step(action)
        return action

class ZeroMCTSPlayer(Player):
    def __init__(self, game: GomokuEnv, policy: ZeroPolicy, itermax=2000):
        super().__init__("ZeroMCTS Player")
        self.game = game
        self.itermax = itermax
        self.policy = policy

    def play(self):
        # Implement MCTS logic here
        mcts = ZeroMCTS(self.game, self.policy) 
        mcts.run(self.itermax)

        num_moves = self.game.move_size # 你需要一个方法来获取当前是第几步
        temperature = 1.0 if num_moves < 30 else 0.0

        # 使用带温度的采样来选择最终动作
        action, probs_for_training = mcts.select_action_with_temperature(temperature)

        self.game.step(action)

        # probs = mcts.root.
        probs = [] 
        for i in range(self.game.board_size ** 2):
            child_visits = mcts.root.children[i].visits if i in mcts.root.children else 0
            probs.append(child_visits / mcts.root.visits if mcts.root.visits > 0 else 0)
        return {
            'action': action, 
            'probs': probs, 
            'state': self.game._get_observation()
        }

class RandomPlayer(Player):
    def __init__(self, game: GomokuEnv):
        super().__init__("Random Player")
        self.game = game

    def play(self):
        valid_actions = self.game.get_valid_actions()
        if not valid_actions:
            return None
        action = random.choice(valid_actions)
        self.game.step(action)

        return action

# %%
if __name__ == "__main__":

    board_size = 9
    game = GomokuEnv(board_size=board_size)
    policy = ZeroPolicy(board_size, device='cpu')
    policy.load_state_dict(torch.load('models/gomoku_zero_lr/policy_step_400.pth'))
    player1 = ZeroMCTSPlayer(game, policy, itermax=800)
    player2 = MCTSPlayer(game, itermax=1000)

    while not game._is_terminal():
        # print(game)
        game.render()
        action1 = player1.play()
        # print(f"{player1.name} played: {action1}")
        
        if game._is_terminal():
            break
        
        action2 = player2.play()
        # print(f"{player2.name} played: {action2}")

    game.render()
    print("Game Over!")

# %%

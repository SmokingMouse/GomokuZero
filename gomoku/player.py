
# %% 
from gomoku.mcts import MCTS, ZeroMCTS
from gomoku.gomoku_env import GomokuEnv, GomokuEnvSimple
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
    def __init__(self, game: GomokuEnv, policy: ZeroPolicy, itermax=2000, device='cpu', eager=False):
        super().__init__("ZeroMCTS Player")
        self.game = game
        self.itermax = itermax
        self.policy = policy
        self.device = device
        self.eager = eager

    def play(self):
        # Implement MCTS logic here
        current_state = self.game._get_observation()# 关键，不能是执行动作后的状态

        if self.eager:
            mcts = ZeroMCTS(self.game, self.policy, device=self.device, dirichlet_alpha=0) 
        else:
            mcts = ZeroMCTS(self.game, self.policy, device=self.device) 

        mcts.run(self.itermax)

        num_moves = self.game.move_size # 你需要一个方法来获取当前是第几步
        if self.eager:
            temperature = 0.0
        else:
            temperature = 1.0 if num_moves < 30 else 0.0

        # 使用带温度的采样来选择最终动作
        action, probs_for_training = mcts.select_action_with_temperature(temperature)

        self.game.step(action)

        # probs = mcts.root.
        # probs = [] 
        # for i in range(self.game.board_size ** 2):
        #     child_visits = mcts.root.children[i].visits if i in mcts.root.children else 0
        #     probs.append(child_visits / mcts.root.visits if mcts.root.visits > 0 else 0)
        return {
            'action': action, 
            'probs': probs_for_training, 
            'state': current_state
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

def self_play(policy, device, itermax=800):
    game = GomokuEnvSimple()
    player1 = ZeroMCTSPlayer(game, policy, itermax=itermax, device=device)
    player2 = ZeroMCTSPlayer(game, policy, itermax=itermax, device=device)

    states = []
    probs = []
    rewards = []

    while not game._is_terminal():
        infos = player1.play()
        states.append(infos['state'])
        probs.append(infos['probs'])

        if game._is_terminal():
            break
        infos = player2.play()
        states.append(infos['state'])
        probs.append(infos['probs'])

    winner = game.winner
    for i in range(len(states)):
        current_player = i % 2 + 1

        if current_player == winner:
            rewards.append(1)
        elif winner == 0:
            rewards.append(0)
        else:
            rewards.append(-1)
    
    print(f"Game over! Winner: {winner}")
    # game.render()

    return {
        'states': states,
        'probs': probs,
        'rewards': rewards,
    }

# %%
if __name__ == "__main__":

    board_size = 9
    game = GomokuEnv(board_size=board_size)
    policy = ZeroPolicy(board_size)
    policy.load_state_dict(torch.load('models/gomoku_zero_ray_dirichlet/policy_step_8400.pth'))
    policy2 = ZeroPolicy(board_size)
    policy2.load_state_dict(torch.load('models/gomoku_zero_ray_dirichlet_800/policy_step_2600.pth'))
    player1 = ZeroMCTSPlayer(game, policy, itermax=400, eager=True)
    # player2 = ZeroMCTSPlayer(game, policy, itermax=1600, eager=True)
    player2 = MCTSPlayer(game, itermax=8000)

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

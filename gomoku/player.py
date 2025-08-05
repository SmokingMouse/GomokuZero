
# %% 
from email.policy import Policy
from gomoku.mcts import MCTS, ZeroMCTS
from gomoku.gomoku_env import GomokuEnv, GomokuEnvSimple
from gomoku.policy import ZeroPolicy
from gomoku.utils import timer
import concurrent.futures
import tqdm
import argparse
import random
import torch
import os
import ray
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
        # self.game = game
        self.itermax = itermax

    def play(self, game):
        # Implement MCTS logic here
        mcts = MCTS(game) 
        action = mcts.run(self.itermax)

        game.step(action)
        return {
            'action': action, 
            'probs': [], 
            'state': []
        }

class ZeroMCTSPlayer(Player):
    def __init__(self, policy: ZeroPolicy, itermax=2000, device='cpu', eager=False):
        super().__init__("ZeroMCTS Player")
        self.itermax = itermax
        self.policy = policy
        self.device = device
        self.eager = eager

    def play(self, game):
        self.policy.eval()
        # Implement MCTS logic here
        current_state = game._get_observation()# 关键，不能是执行动作后的状态

        if self.eager:
            mcts = ZeroMCTS(game, self.policy, device=self.device, dirichlet_alpha=0) 
        else:
            mcts = ZeroMCTS(game, self.policy, device=self.device) 

        mcts.run(self.itermax)

        num_moves = game.move_size # 你需要一个方法来获取当前是第几步
        if self.eager:
            temperature = 0.0
        else:
            temperature = 1.0 if num_moves < 10 else 0.0

        # 使用带温度的采样来选择最终动作
        action, probs_for_training = mcts.select_action_with_temperature(temperature)

        game.step(action)

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
    # game = GomokuEnvSimple()
    player1 = ZeroMCTSPlayer(policy, itermax=itermax, device=device)
    player2 = ZeroMCTSPlayer(policy, itermax=itermax, device=device)

    winner, infos = play_one_game(player1, player2)
    print(f"Game over! Winner: {winner}")
    return infos

def play_one_game(player1, player2, game: GomokuEnvSimple = None):
    states = []
    probs = []
    rewards = []

    if game is None:
        env = GomokuEnvSimple()
    else:
        env = game

    while not env._is_terminal():
        infos = player1.play(env)
        states.append(infos['state'])
        probs.append(infos['probs'])

        # env.render()
        if env._is_terminal():
            break
        infos = player2.play(env)
        states.append(infos['state'])
        probs.append(infos['probs'])
        # env.render()

    winner = env.winner
    for i in range(len(states)):
        current_player = i % 2 + 1

        if current_player == winner:
            rewards.append(1)
        elif winner == 0:
            rewards.append(0)
        else:
            rewards.append(-1)
    
    return winner, {
        'states': states,
        'probs': probs,
        'rewards': rewards,
    }


@ray.remote
def play_game_worker(player1, player2):
    return play_one_game(player1, player2)

@timer
def arena(player1, player2, games = 1):
    ray.init(os.cpu_count()-4)
    player1_win_count = 0
    for game in range(games):
        # env = GomokuEnvSimple()
        player1_turn = random.choice([1, 2])
        if player1_turn == 1:
            winner, _ = play_one_game(player1, player2)
        else:
            winner, _ = play_one_game(player2, player1)

        if winner == player1_turn:
            player1_win_count += 1
    ray.shutdown()
    return player1_win_count / games

@timer
def arena_parallel(player1_main, player2_main, games=100, num_cpus=None):
    if num_cpus is None:
        num_cpus = os.cpu_count()

    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
    
    print(f"Starting parallel arena with {games} games on {num_cpus} CPUs...")
    
    # 将大的玩家对象放入 Ray 对象存储
    p1_ref = ray.put(player1_main)
    p2_ref = ray.put(player2_main)
    
    task_refs = []
    player1_turn_flags = [] # 记录每场比赛是不是 player1_main 先手

    for _ in range(games):
        # 随机决定谁先手
        if random.choice([True, False]):
            # player1_main 执黑先手
            first_player_ref = p1_ref
            second_player_ref = p2_ref
            player1_turn_flags.append(True)
        else:
            # player2_main 执黑先手
            first_player_ref = p2_ref
            second_player_ref = p1_ref
            player1_turn_flags.append(False)
            
        # 启动远程任务，play_one_game 总是认为第一个参数是先手
        task = play_game_worker.remote(first_player_ref, second_player_ref)
        task_refs.append(task)
        
    # 等待并收集所有比赛结果
    results = ray.get(task_refs)
    
    # 提取所有获胜者编号 (1=先手胜, 2=后手胜, 0=平局)
    raw_winners = [res[0] for res in results]
    
    # 统计结果
    player1_wins = 0
    player2_wins = 0
    draws = 0
    
    for i in range(games):
        winner = raw_winners[i]
        is_p1_turn = player1_turn_flags[i]
        
        if winner == 0:
            draws += 1
        elif winner == 1: # 先手获胜
            if is_p1_turn:
                player1_wins += 1
            else:
                player2_wins += 1
        elif winner == 2: # 后手获胜
            if is_p1_turn:
                player2_wins += 1
            else:
                player1_wins += 1
                
    assert player1_wins + player2_wins + draws == games

    player1_win_rate = player1_wins / games
    player2_win_rate = player2_wins / games
    draw_rate = draws / games

    print("Arena finished!")
    print(f"Player 1 wins: {player1_wins} ({player1_win_rate:.2%})")
    print(f"Player 2 wins: {player2_wins} ({player2_win_rate:.2%})")
    print(f"Draws: {draws} ({draw_rate:.2%})")

    return {
        'player1_win_rate': player1_win_rate,
        'player2_win_rate': player2_win_rate,
        'draw_rate': draw_rate,
    }
# %%
# game = GomokuEnvSimple()
# policy = ZeroPolicy(board_size=9)
# policy.load_state_dict(torch.load('models/gomoku_zero_freqency/policy_step_2500.pth'))
# _ = self_play(policy, 'cpu', 400)

# %%
if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument('--games', type=int, default=20)
    argparse.add_argument('--itermax', type=int, default=400)
    # argparse.add_argument('--eager', action='store_true')
    argparse.add_argument('--device', type=str, default='cpu')
    argparse.add_argument('--model1', type=str, default='gomoku_zero_resnet/policy_step_2500.pth')
    argparse.add_argument('--model2', type=str, default='gomoku_zero_resnet_play30/policy_step_4000.pth')
    argparse.add_argument('--pure_mcts_iter', type=int, default=0)

    args = argparse.parse_args()

    # ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = '/home/zhangpeng.pada/GomokuZero/gomoku/models'

    board_size = 9
    game = GomokuEnv(board_size=board_size)
    policy = ZeroPolicy(board_size)
    policy.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.model1)))
    player1 = ZeroMCTSPlayer(policy, itermax=args.itermax, eager=False)

    if args.pure_mcts_iter > 0:
        player2 = MCTSPlayer(game, itermax=args.pure_mcts_iter)
    else:
        policy2 = ZeroPolicy(board_size)
        policy2.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.model2)))
        player2 = ZeroMCTSPlayer(policy2, itermax=args.itermax, eager=False)

    win_rate = arena_parallel(player1, player2, games=args.games)
    print(f"Win rate: {win_rate}")
# %%

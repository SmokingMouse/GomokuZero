
# %% 
from email.policy import Policy
import time
import rich
from tkinter import NO
from gomoku.mcts import MCTS, WrongZeroMCTS
from gomoku.zero_mcts import ZeroMCTS
from gomoku.gomoku_env import GomokuEnv, GomokuEnvSimple
from gomoku.policy import ZeroPolicy
from gomoku.utils import timer
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
    def __init__(self, game: GomokuEnv, itermax=200):
        super().__init__("MCTS Player")
        # self.game = game
        self.itermax = itermax

    def play(self, game, *args, **kwargs):
        # Implement MCTS logic here
        mcts = MCTS(game) 
        action = mcts.run(self.itermax)

        game.step(action)
        return {
            'action': action, 
            'probs': [], 
            'state': []
        }

class IneffectiveZeroMCTSPlayer(Player):
    def __init__(self, policy: ZeroPolicy, itermax=2000, device='cpu', eager=False):
        super().__init__("Ineffective ZeroMCTS Player")
        self.itermax = itermax
        self.policy = policy
        self.device = device
        self.eager = eager

    # @timer
    def play(self, game, *args, **kwargs):
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
        action, probs_for_training = mcts.select_action_with_temperature(temperature, 7)

        game.step(action)

        return {
            'action': action, 
            'probs': probs_for_training, 
            'state': current_state
        }

class WrongZeroMCTSPlayer(Player):
    def __init__(self, policy: ZeroPolicy, itermax=200, device='cpu', eager=False):
        super().__init__("Wrong ZeroMCTS Player")
        self.itermax = itermax
        self.policy = policy
        self.device = device
        self.eager = eager

    # @timer
    def play(self, game, *args, **kwargs):
        self.policy.eval()
        # Implement MCTS logic here
        current_state = game._get_observation()# 关键，不能是执行动作后的状态

        if self.eager:
            mcts = WrongZeroMCTS(game, self.policy, device=self.device, dirichlet_alpha=0) 
        else:
            mcts = WrongZeroMCTS(game, self.policy, device=self.device) 

        mcts.run(self.itermax)

        num_moves = game.move_size # 你需要一个方法来获取当前是第几步
        if self.eager:
            temperature = 0.0
        else:
            temperature = 1.0 if num_moves < 10 else 0.0

        # 使用带温度的采样来选择最终动作
        action, probs_for_training = mcts.select_action_with_temperature(temperature)

        game.step(action)

        return {
            'action': action, 
            'probs': probs_for_training, 
            'state': current_state
        }

class ZeroMCTSPlayer(Player):
    def __init__(self, policy: ZeroPolicy, device = 'cpu'):
        super().__init__("ZeroMCTS Player")
        self.policy = policy
        self.device = device

    # @timer
    def play(self, game, *args, **kwargs):
        self.policy.eval()
        # Implement MCTS logic here
        current_state = game._get_observation()# 关键，不能是执行动作后的状态

        mcts = kwargs.get('mcts')
        itermax = kwargs.get('itermax')

        if mcts is None or itermax is None:
            raise ValueError("MCTS instance and itermax must be provided in kwargs.")
        mcts.run(itermax)

        num_moves = game.move_size # 你需要一个方法来获取当前是第几步

        eager = kwargs.get('eager', False)

        temperature_moves = kwargs.get('temperature_moves', 10)

        if eager:
            temperature = 0.0
        else:
            temperature = 1.0 if num_moves < temperature_moves else 0.0

        # 使用带温度的采样来选择最终动作
        action, probs_for_training = mcts.select_action_with_temperature(temperature)

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

def self_play(policy, device, board_size, itermax=200):
    player1 = ZeroMCTSPlayer(policy, device=device)
    player2 = ZeroMCTSPlayer(policy, device=device)

    winner, infos = play_one_game(
        player1, player2, board_size=board_size,
        itermax=itermax, eager=False
    )
    print(f"Game over! Winner: {winner}")
    return infos

@timer
def play_one_game(player1, player2, board_size: int, 
                  game: GomokuEnv = None,
                  render=False, itermax=200, eager=False):
    states = []
    probs = []
    rewards = []

    if game is None:
        env = GomokuEnv(board_size)
    else:
        env = game
    
    mcts1 = ZeroMCTS(env, player1.policy, device=player1.device)
    mcts2 = ZeroMCTS(env, player2.policy, device=player2.device)

    while not env._is_terminal():
        infos = player1.play(env, **{
            'mcts': mcts1, 
            'itermax': itermax, 
            'eager': eager
        }) 
        states.append(infos['state'])
        probs.append(infos['probs'])

        action1 = infos['action']
        env.step(action1)
        mcts1.update_root(action1)
        mcts2.update_root(action1)
        if render:
            env.render()
        if env._is_terminal():
            break
        infos = player2.play(env, **{
            'mcts': mcts2, 
            'itermax': itermax, 
            'eager': eager
        })
        states.append(infos['state'])
        probs.append(infos['probs'])

        action2 = infos['action']
        env.step(action2)
        mcts1.update_root(action2)
        mcts2.update_root(action2)


        if render:
            env.render()

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
        'env': env
    }

@ray.remote
class ArenaWorker:
    def __init__(self, player1_model: ZeroPolicy, player2_model: ZeroPolicy, board_size: int, itermax: int = 200, eager: bool = False):
        # 在 Actor 创建时，一次性初始化 Player 对象
        # 这个包含了神经网络的反序列化过程，只会在 Actor 启动时发生一次！
        self.player1 = ZeroMCTSPlayer(player1_model)
        self.player2 = ZeroMCTSPlayer(player2_model)
        self.board_size = board_size
        self.itermax = itermax
        self.eager = eager

    def run_game(self, player1_starts:bool):
        if player1_starts:
            first_player, second_player = self.player1, self.player2
        else:
            first_player, second_player = self.player2, self.player1
            
        s = play_one_game(
            first_player, 
            second_player, 
            board_size=self.board_size, 
            itermax=self.itermax, 
            eager=self.eager 
        )
        return s

@timer
def arena_parallel(policy1, policy2, board_size, num_cpus, games=100, eager=False, itermax=200):
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)
    
    print(f"Starting parallel arena with {games} games on {num_cpus} CPUs...")
    p1_ref = ray.put(policy1)
    p2_ref = ray.put(policy2)

    actor_pool = [ArenaWorker.remote(p1_ref, p2_ref, board_size=board_size, eager=eager, itermax=itermax) for _ in range(num_cpus)]
    pool_size = len(actor_pool)

    task_refs = []
    player1_turn_flags = []

    for i in range(games):
        player1_starts = random.choice([True, False])
        player1_turn_flags.append(player1_starts)
        
        actor = actor_pool[i % pool_size]
        
        task = actor.run_game.remote(player1_starts)
        task_refs.append(task)

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
        
        if winner == 0 or winner is None:  # 平局或无胜者
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
# policy.load_state_dict(torch.load('models/gomoku_zero_multisteplr/policy_step_15000.pth'))
# infos = self_play(policy, 'cpu', 200)
# infos['env'].render()


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
    argparse.add_argument('--root', type=str, default='/home/zhangpeng.pada/GomokuZero/gomoku/models')

    args = argparse.parse_args()

    # ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
    ROOT_PATH = args.root

    board_size = 9
    game = GomokuEnv(board_size=board_size)
    policy = ZeroPolicy(board_size)
    policy.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.model1)))
    # player1 = ZeroMCTSPlayer(policy, itermax=args.itermax, eager=False)

    # if args.pure_mcts_iter > 0:
        # player2 = MCTSPlayer(game, itermax=args.pure_mcts_iter)
    # else:
    policy2 = ZeroPolicy(board_size)
    policy2.load_state_dict(torch.load(os.path.join(ROOT_PATH, args.model2)))
    # player2 = ZeroMCTSPlayer(policy2, itermax=args.itermax, eager=False)

    win_rate = arena_parallel(policy, policy2, games=args.games, board_size=9, num_cpus=os.cpu_count()//2)
    print(f"Win rate: {win_rate}")
# %%

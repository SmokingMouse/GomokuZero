# %%
from gomoku.mcts import MCTS
from gomoku.zero_mcts import ZeroMCTS
from gomoku.light_zero_mcts import LightZeroMCTS
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
from gomoku.utils import timer
import argparse
import random
import torch
import os
import ray
import importlib
import numpy as np
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
    def __init__(self, itermax=200):
        super().__init__("MCTS Player")
        # self.game = game
        self.itermax = itermax

    def play(self, game, *args, **kwargs):
        # Implement MCTS logic here
        mcts = MCTS(game)
        action = mcts.run(self.itermax)

        # game.step(action)
        return {"action": action, "probs": [], "state": []}


class ZeroMCTSPlayer(Player):
    def __init__(self, policy: ZeroPolicy, device="cpu"):
        super().__init__("ZeroMCTS Player")
        self.policy = policy
        self.device = device

    # @timer
    def play(self, game, *args, **kwargs):
        self.policy.eval()
        # Implement MCTS logic here
        current_state = game._get_observation()  # 关键，不能是执行动作后的状态

        mcts = kwargs.get("mcts")
        itermax = kwargs.get("itermax")

        if mcts is None or itermax is None:
            raise ValueError("MCTS instance and itermax must be provided in kwargs.")
        mcts.run(game, itermax)

        num_moves = game.move_size  # 你需要一个方法来获取当前是第几步

        eager = kwargs.get("eager", False)

        temperature_moves = kwargs.get("temperature_moves", 10)

        if eager:
            temperature = 0.0
        else:
            temperature = 1.0 if num_moves < temperature_moves else 0.0

        # 使用带温度的采样来选择最终动作
        action, probs_for_training = mcts.select_action_with_temperature(temperature)

        return {"action": action, "probs": probs_for_training, "state": current_state}


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


def make_policy_fn(policy: ZeroPolicy, device: str):
    policy.eval()

    def policy_fn(obs: np.ndarray):
        x = torch.from_numpy(obs).reshape(1, -1).float().to(device)
        with torch.no_grad():
            policy_logits, value = policy(x)
        probs = policy_logits.squeeze(0).cpu().numpy()
        return probs, float(value.item())

    return policy_fn


def self_play(
    policy,
    device,
    board_size,
    itermax=200,
    use_rs_mcts: bool = False,
    branch_ratio: float = 0.2,
):
    player1 = ZeroMCTSPlayer(policy, device=device)
    player2 = ZeroMCTSPlayer(policy, device=device)

    env = GomokuEnv(board_size)

    winner, infos = play_one_game(
        player1,
        player2,
        board_size=board_size,
        game=env,
        itermax=itermax,
        eager=False,
        use_rs_mcts=use_rs_mcts,
    )
    games = [infos]

    if (
        branch_ratio > 0
        and infos.get("actions")
        and random.random() < branch_ratio
        and len(infos["actions"]) > 3
    ):
        start_idx = random.randint(1, len(infos["actions"]) - 2)
        branch_env = GomokuEnv(board_size)
        for action in infos["actions"][:start_idx]:
            branch_env.step(action)
            if branch_env.done:
                break
        if not branch_env.done:
            branch_winner, branch_infos = play_one_game(
                player1,
                player2,
                board_size=board_size,
                game=branch_env,
                itermax=itermax,
                eager=False,
                use_rs_mcts=use_rs_mcts,
                forbidden_first_action=infos["actions"][start_idx],
            )
            games.append(branch_infos)
            print(f"Branch game over! Winner: {branch_winner}")

    print(f"Game over! Winner: {winner}")
    return games


@timer
def play_one_game(
    player1,
    player2,
    board_size: int,
    game: GomokuEnv = None,
    render=False,
    itermax=200,
    eager=False,
    MCTS=LightZeroMCTS,
    use_rs_mcts: bool = False,
    temperature_moves: int = 8,
    use_dirichlet: bool = True,
    forbidden_first_action: int | None = None,
):
    states = []
    probs = []
    rewards = []
    actions = []
    players_turns = []

    if game is None:
        env = GomokuEnv(board_size)
    else:
        env = game

    if use_rs_mcts:
        zero_mcts_rs = importlib.import_module("zero_mcts_rs")
        if not hasattr(zero_mcts_rs, "LightZeroMCTS"):
            raise RuntimeError(
                "zero_mcts_rs module loaded, but LightZeroMCTS is missing. "
                "Run maturin develop in zero_mcts_rs first."
            )
        policy_fn1 = make_policy_fn(player1.policy, player1.device)
        policy_fn2 = make_policy_fn(player2.policy, player2.device)
        mcts1 = zero_mcts_rs.LightZeroMCTS(
            board_size=board_size,
            puct=2.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.15,
        )
        mcts2 = zero_mcts_rs.LightZeroMCTS(
            board_size=board_size,
            puct=2.0,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.15,
        )
    else:
        mcts1 = MCTS(player1.policy, device=player1.device)
        mcts2 = MCTS(player2.policy, device=player2.device)

    forbidden_used = False

    while not env._is_terminal():
        current_player = env.current_player
        forbidden_actions = None
        if not forbidden_used and forbidden_first_action is not None:
            forbidden_actions = [forbidden_first_action]

        if use_rs_mcts:
            current_state = env._get_observation()
            board_flat = env.board.astype(np.int8).reshape(-1)
            if current_player == 1:
                mcts1.run(
                    board_flat,
                    env.current_player,
                    policy_fn1,
                    iterations=itermax,
                    use_dirichlet=use_dirichlet,
                    move_size=env.move_size,
                    last_action=env.last_action,
                )
                num_moves = env.move_size
                temperature = (
                    0.0 if eager else (1.0 if num_moves < temperature_moves else 0.0)
                )
                action, probs_for_training = mcts1.select_action_with_temperature(
                    temperature, None, forbidden_actions
                )
            else:
                mcts2.run(
                    board_flat,
                    env.current_player,
                    policy_fn2,
                    iterations=itermax,
                    use_dirichlet=use_dirichlet,
                    move_size=env.move_size,
                    last_action=env.last_action,
                )
                num_moves = env.move_size
                temperature = (
                    0.0 if eager else (1.0 if num_moves < temperature_moves else 0.0)
                )
                action, probs_for_training = mcts2.select_action_with_temperature(
                    temperature, None, forbidden_actions
                )
        else:
            player = player1 if current_player == 1 else player2
            mcts = mcts1 if current_player == 1 else mcts2
            infos = player.play(
                env,
                **{
                    "mcts": mcts,
                    "itermax": itermax,
                    "eager": eager,
                    "temperature_moves": temperature_moves,
                    "forbidden_actions": forbidden_actions,
                },
            )
            current_state = infos["state"]
            probs_for_training = infos["probs"]
            action = infos["action"]

        if forbidden_actions is not None:
            forbidden_used = True

        states.append(current_state)
        probs.append(probs_for_training)
        actions.append(action)
        players_turns.append(current_player)

        env.step(action)
        mcts1.step(action)
        mcts2.step(action)
        if render:
            env.render()

    winner = env.winner
    for i, current_player in enumerate(players_turns):
        if current_player == winner:
            rewards.append(1)
        elif winner == 0:
            rewards.append(0)
        else:
            rewards.append(-1)

    return winner, {
        "states": states,
        "probs": probs,
        "rewards": rewards,
        "actions": actions,
        "env": env,
    }


@ray.remote
class ArenaWorker:
    def __init__(
        self,
        player1_model: ZeroPolicy,
        player2_model: ZeroPolicy,
        board_size: int,
        itermax: int = 200,
        eager: bool = False,
    ):
        # 在 Actor 创建时，一次性初始化 Player 对象
        # 这个包含了神经网络的反序列化过程，只会在 Actor 启动时发生一次！
        self.player1 = ZeroMCTSPlayer(player1_model)
        self.player2 = ZeroMCTSPlayer(player2_model)
        self.board_size = board_size
        self.itermax = itermax
        self.eager = eager

    def run_game(self, player1_starts: bool):
        if player1_starts:
            first_player, second_player = self.player1, self.player2
        else:
            first_player, second_player = self.player2, self.player1

        s = play_one_game(
            first_player,
            second_player,
            board_size=self.board_size,
            itermax=self.itermax,
            eager=self.eager,
        )
        return s


@timer
def arena_parallel(
    policy1,
    policy2,
    board_size,
    num_cpus,
    games=100,
    eager=False,
    itermax=200,
    MCTS=ZeroMCTS,
):
    # 初始化 Ray
    if not ray.is_initialized():
        ray.init(num_cpus=num_cpus)

    print(f"Starting parallel arena with {games} games on {num_cpus} CPUs...")
    p1_ref = ray.put(policy1)
    p2_ref = ray.put(policy2)

    actor_pool = [
        ArenaWorker.remote(
            p1_ref, p2_ref, board_size=board_size, eager=eager, itermax=itermax
        )
        for _ in range(num_cpus)
    ]
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
        elif winner == 1:  # 先手获胜
            if is_p1_turn:
                player1_wins += 1
            else:
                player2_wins += 1
        elif winner == 2:  # 后手获胜
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
        "player1_win_rate": player1_win_rate,
        "player2_win_rate": player2_win_rate,
        "draw_rate": draw_rate,
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
    argparse.add_argument("--games", type=int, default=20)
    argparse.add_argument("--itermax", type=int, default=400)
    # argparse.add_argument('--eager', action='store_true')
    argparse.add_argument("--device", type=str, default="cpu")
    argparse.add_argument(
        "--model1", type=str, default="gomoku_zero_resnet/policy_step_2500.pth"
    )
    argparse.add_argument(
        "--model2", type=str, default="gomoku_zero_resnet_play30/policy_step_4000.pth"
    )
    argparse.add_argument("--pure_mcts_iter", type=int, default=0)
    argparse.add_argument(
        "--root", type=str, default="/home/zhangpeng.pada/GomokuZero/gomoku/models"
    )

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

    win_rate = arena_parallel(
        policy, policy2, games=args.games, board_size=9, num_cpus=os.cpu_count() // 2
    )
    print(f"Win rate: {win_rate}")
# %%

#%%
from tkinter.tix import Tree
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
import torch.nn.functional as F
import torch
import random
import math
from abc import ABC, abstractmethod
import numpy as np

class Strategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def Choose(self, actions):
        pass

class RandomStrategy(Strategy):
    def __init__(self):
        super().__init__()
    
    def name(self):
        return 'random'
    
    def Choose(self, actions):
        return random.choice(actions)

random_strategy = RandomStrategy()
random_strategy.name()

import math
import random
import numpy as np
from abc import ABC, abstractmethod

# 假设 GomokuEnv 和 Strategy, RandomStrategy 类已经定义好了
# from gomoku.gomoku_env import GomokuEnv
# from your_strategy_file import Strategy, RandomStrategy
class Strategy(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def Choose(self, actions):
        pass

class RandomStrategy(Strategy):
    def __init__(self):
        super().__init__()
    
    def name(self):
        return 'random'
    
    def Choose(self, actions):
        return random.choice(actions)

class TreeNode:
    def __init__(self, env: 'GomokuEnv'):
        self.env = env
        self.visits = 0
        self.value_sum = 0.0  # 使用浮点数，并改名为 value_sum 更清晰
        self.children = {}
        self.parent = None

    def add_child(self, action, child_node):
        child_node.parent = self
        self.children[action] = child_node

    def update(self, value_from_child_perspective):
        """
        更新节点。接收的 value 是从子节点玩家视角看的。
        我们的 value_sum 存储的是从当前节点玩家视角看的累计价值。
        所以需要取反。
        """
        self.visits += 1
        self.value_sum -= value_from_child_perspective

    @property
    def q_value(self):
        """返回从当前节点玩家视角看的平均价值。"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def is_fully_expanded(self):
        """检查节点是否已完全扩展。"""
        valid_actions = self.env.get_valid_actions()
        return len(self.children) == len(valid_actions)


class MCTS:
    def __init__(self, strategy=RandomStrategy(), c=1.41):
        self.strategy = strategy
        self.c = c  # UCT 的探索常数
        self.env2node = {}

    def hash(self, env: 'GomokuEnv'): 
        return env.board.tobytes()

    def run(self, env, iterations):
        """运行 MCTS 并返回最佳动作。"""
        hash_key = self.hash(env)
        if hash_key not in self.env2node: 
            self.env2node[hash_key] = TreeNode(env.clone())
        
        root = self.env2node[hash_key]
        root.parent = None # 确保每次运行的根节点没有父节点

        for _ in range(iterations):
            # 1. Selection: 找到一个需要扩展的节点
            node_to_explore = self._select(root)
            
            # 2. Expansion: 如果它不是终局，就扩展它
            if not node_to_explore.env._is_terminal():
                node_to_explore = self._expansion(node_to_explore)
            
            # 3. Simulation: 从新创建或选中的节点开始模拟
            simulation_result_value = self._simulate(node_to_explore)
            
            # 4. Backpropagation: 从模拟开始的节点向上回溯
            self._backpropagation(node_to_explore, simulation_result_value)
        
        return self._best_action(root)
    
    def _select(self, node: TreeNode):
        """使用 UCT 选择节点，直到遇到未完全扩展或终止的节点。"""
        current = node
        while not current.env._is_terminal() and current.is_fully_expanded():
            if not current.children:
                # 如果一个完全扩展的节点没有子节点，说明是平局等情况
                return current
            current = max(current.children.values(), key=lambda child: self._uct_value(child))
        return current

    def _expansion(self, node: TreeNode):
        """从给定节点扩展出一个新的子节点。"""
        valid_actions = node.env.get_valid_actions()
        expanded_actions = set(node.children.keys())
        
        # 找出所有未被探索过的动作
        unexplored_actions = [a for a in valid_actions if a not in expanded_actions]
        
        if not unexplored_actions:
            # 理论上，如果 is_fully_expanded() 是 False，这里不应该为空
            return node
            
        # 从未探索的动作中选择一个来扩展
        action = self.strategy.Choose(unexplored_actions)
        
        child_env = node.env.clone()
        child_env.step(action)
        
        # 检查这个新状态是否已经存在于我们的全局字典中
        child_hash = self.hash(child_env)
        if child_hash in self.env2node:
            child_node = self.env2node[child_hash]
        else:
            child_node = TreeNode(child_env)
            self.env2node[child_hash] = child_node

        node.add_child(action, child_node)
        return child_node

    def _simulate(self, node: TreeNode):
        """
        模拟随机游戏并返回从 node 玩家视角看的价值。
        返回值: 1 for win, -1 for lose, 0 for draw.
        """
        env = node.env.clone()
        while not env._is_terminal():
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = self.strategy.Choose(valid_actions)
            env.step(action)
        
        winner = env.winner
        if winner == 0:
            return 0.0
        
        # 如果最终的赢家是 node 节点的当前玩家，那么从 node 视角看，价值是 +1
        # 否则是 -1
        return 1.0 if winner == node.env.current_player else -1.0

    def _backpropagation(self, node: TreeNode, value: float):
        """
        反向传播价值。
        
        Args:
            node: 模拟开始的节点。
            value: 从 node 玩家视角看的模拟结果价值。
        """
        current = node
        while current is not None:
            # update 方法接收的是从子节点视角看的价值，
            # 而我们回传的 value 是从父节点视角看的。
            # 所以，在调用 update 前，需要将 value 取反。
            current.update(-value)
            # value 本身也需要取反，因为我们向上移动了一层，玩家视角反转。
            value *= -1
            current = current.parent
    
    def _best_action(self, root: TreeNode):
        """根据访问次数返回最佳动作。"""
        if not root.children:
            # 如果没有子节点（比如迭代次数为0或1），随机选择一个
            return self.strategy.Choose(root.env.get_valid_actions())
        return max(root.children.items(), key=lambda item: item[1].visits)[0]
    
    def _uct_value(self, child: TreeNode):
        """
        计算子节点的 UCT 值，站在父节点的视角。
        """
        if child.visits == 0:
            # 优先探索未被访问的节点
            return float('inf')
        
        # exploitation term: 子节点的平均价值，但要转换到父节点的视角
        # child.q_value 是从 child 玩家视角看的，所以要取反
        exploitation = -child.q_value
        
        # exploration term
        exploration = self.c * math.sqrt(math.log(child.parent.visits) / child.visits)
        
        return exploitation + exploration
    

#%% 
# class TreeNode:
#     def __init__(self, env: GomokuEnv):
#         self.env = env
#         self.visits = 0
#         self.wins = 0
#         self.children = {}
#         self.parent = None
#         self.action_prob = None

#     def add_child(self, action, child_node):
#         child_node.parent = self
#         self.children[action] = child_node

#     def update(self, result):
#         self.visits += 1

#         player = result[0]
#         if player == self.env.current_player:
#             self.wins += result[1]  # result[1] is the value from the simulation
#         # elif player == 3 - self.env.current_player:  # Assuming 1 and 2 are players
#             # self.wins -= result[1]
    
#     def is_fully_expanded(self):
#         """Check if the node is fully expanded"""
#         valid_actions = self.env.get_valid_actions()
#         return len(self.children) == len(valid_actions)

# class MCTS:
#     def __init__(self, strategy=RandomStrategy(), c=1.41, puct = 5):
#         self.strategy = strategy
#         self.c = c  # Exploration constant for UCT
#         self.puct = puct
#         # self.root = TreeNode(env.clone())
#         self.env2node = {}
    
#     def hash(self, env: GomokuEnv): 
#         return env.board.tobytes()
    
#     def find_root(self, env): 
#         return self.env2node.get(self.hash(env), None)

#     def run(self, env, iterations):
#         """Run MCTS and return the best action"""
#         hash_key = self.hash(env)
#         if hash_key not in self.env2node: 
#             self.env2node[hash_key] = TreeNode(env.clone())
#         self.root = self.env2node[hash_key]
#         self.root.parent = None

#         for _ in range(iterations):
#             node = self._select(self.root)
            
#             if not node.env._is_terminal() and not node.is_fully_expanded():
#                 node = self._expansion(node)
            
#             result = self._simulate(node)
#             self._backpropagation(node, result)
        
#         return self._best_action(self.root)
    
#     def _select(self, node: TreeNode):
#         """Select node using UCT"""
#         current = node
#         while not current.env._is_terminal() and current.is_fully_expanded():
#             strategy_name = self.strategy.name()
#             if strategy_name == 'random':
#                 current = max(current.children.values(), key=lambda child: self._uct_value(child))
#             elif strategy_name == 'zero':
#                 current = max(current.children.values(), key=lambda child: self._puct_value(child))
#             else:
#                 raise ValueError("unsupported strategy!")

#         return current

#     def _expansion(self, node):
#         """Expand the node"""
#         valid_actions = node.env.get_valid_actions()
#         expanded_actions = set(node.children.keys())
#         available_actions = [a for a in valid_actions if a not in expanded_actions]
        
#         if not available_actions:
#             return node
            
#         action = self.strategy.Choose(available_actions)
#         child_env = node.env.clone()
#         child_env.step(action)
        
#         child_node = TreeNode(child_env)
#         node.add_child(action, child_node)
#         self.env2node[self.hash(child_env)] = child_node
#         return child_node

#     def _simulate(self, node):
#         """Simulate a random game"""
#         env = node.env.clone()
#         while not env._is_terminal():
#             valid_actions = env.get_valid_actions()
#             if not valid_actions:
#                 break
#             action = self.strategy.Choose(valid_actions)
#             env.step(action)
#         return (env.winner, 1)

#     def _backpropagation(self, node, result):
#         """Backpropagate the result"""
#         current = node
#         while current is not None:
#             current.update(result)
#             current = current.parent
    
#     def _best_action(self, root: TreeNode):
#         """Return the best action based on visit count"""
#         if not root.children:
#             return self.strategy.Choose(root.env.get_valid_actions())
#         return max(root.children.items(), key=lambda item: item[1].visits)[0]
    
#     def _uct_value(self, child: TreeNode):
#         """Calculate UCT value"""
#         if child.visits == 0:
#             return float('inf')
        
#         exploitation = child.wins / child.visits
#         exploration = self.c * math.sqrt(math.log(child.parent.visits) / child.visits)
        
#         # expoloration = self.c * P_a(Ation) * (sqrt(all visition)  / 1 + N(s, a)
        
#         # For two-player games, we need to negate for the opponent's perspective
#         if child.env.current_player != self.root.env.current_player:
#             return -(exploitation) + exploration
#         else:
#             return exploitation + exploration

class WrongZeroMCTS:
    def __init__(self, env: GomokuEnv, policy: ZeroPolicy, puct=5, device='cpu',
                dirichlet_alpha = 0.3,
                dirichlet_epsilon=0.25):
        # Policy is evaluation network.
        self.env = env
        self.policy = policy
        self.puct = puct 
        self.root = TreeNode(env.clone())
        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
    
    def run(self, iterations):
        """Run MCTS with PUCT using the policy network"""
        if self.dirichlet_alpha > 0:
            self._apply_dirichlet_noise_to_root()
        
        # for _ in range(iterations):
        #     leaf_node = self._select(self.root)
        #     value = self._expand_and_evaluate(leaf_node)
        #     self._backpropagation(leaf_node, value)
        # return self._best_action(self.root)
        for _ in range(iterations):
            node = self._select(self.root)
            
            if not node.env._is_terminal() and not node.is_fully_expanded():
                node = self._expansion(node)
            
            result = self._simulate(node)
            self._backpropagation(node, result)
        
        return self._best_action(self.root) 
    
    def _apply_dirichlet_noise_to_root(self):
        """为根节点的策略概率添加噪声"""
        if self.root.action_prob is None:
            # 如果根节点还没有被评估过，先评估一次
            self._compute_policy_and_value(self.root)

        valid_actions = self.root.env.get_valid_actions()
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions))
        
        for i, action in enumerate(valid_actions):
            self.root.action_prob[action] = \
                (1 - self.dirichlet_epsilon) * self.root.action_prob[action] + \
                self.dirichlet_epsilon * noise[i]
    
    def _compute_policy_and_value(self, node: TreeNode):
        """统一的网络调用函数"""
        # 如果已经计算过，直接返回（尽管在当前流程下不会发生）
        if node.action_prob is not None:
            return node.action_prob, node.q_value

        obs = node.env._get_observation()
        torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        valid_actions_tensor = torch.tensor(node.env.get_valid_actions())
        
        with torch.no_grad():
            policy_logits, value_tensor = self.policy(torch_x)
            
        # Mask invalid actions
        mask = torch.full_like(policy_logits, -1e8)
        mask[0, valid_actions_tensor] = 0.0
        masked_logits = policy_logits + mask
        
        # Convert to probabilities
        action_prob_tensor = torch.softmax(masked_logits, dim=1)[0]
        node.action_prob = {
            action: action_prob_tensor[action].item() for action in valid_actions_tensor.tolist()
        }
        
        return node.action_prob, value_tensor.item()
    
    def _select(self, node: TreeNode):
        """Select node using PUCT"""
        current = node
        while not current.env._is_terminal() and current.is_fully_expanded():
            if current.action_prob is None:
                obs = current.env._get_observation()
                torch_x = torch.from_numpy(obs).unsqueeze(0).float()
                valid_actions_tensor = torch.tensor(current.env.get_valid_actions())
                
                torch_x = torch_x.to(device=self.device)
                with torch.no_grad():
                    policy_logits, value = self.policy(torch_x)
                    
                # Mask invalid actions
                mask = torch.zeros_like(policy_logits)
                mask[0, valid_actions_tensor] = 1.0
                masked_logits = policy_logits * mask + (mask - 1) * 1e8
                
                # Convert to probabilities
                action_prob = torch.softmax(masked_logits, dim=1)[0]
                current.action_prob = {i: action_prob[i].item() for i in valid_actions_tensor.tolist()}

            current = max([
                (
                    child, 
                    self._puct_value(child, action_prob=current.action_prob[action])
                ) for action, child in current.children.items()
            ], key=lambda x: x[1])[0]
        return current
    
    def _expansion(self, node: TreeNode):
        """Expand the node"""
        valid_actions = node.env.get_valid_actions()
        expanded_actions = set(node.children.keys())
        available_actions = [a for a in valid_actions if a not in expanded_actions]
        
        if not available_actions:
            return node
            
        if node.action_prob is None:
            obs = node.env._get_observation()
            torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
            valid_actions_tensor = torch.tensor(node.env.get_valid_actions())
            
            # Get policy and value from network
            with torch.no_grad():
                policy_logits, value = self.policy(torch_x)
                
            # Mask invalid actions
            mask = torch.zeros_like(policy_logits)
            mask[0, valid_actions_tensor] = 1.0
            masked_logits = policy_logits * mask + (mask - 1) * 1e8
            
            # Convert to probabilities
            action_prob = torch.softmax(masked_logits, dim=1)[0]
            node.action_prob = {i: action_prob[i].item() for i in valid_actions_tensor.tolist()}

        # Select action from available actions using policy probabilities
        action = max(available_actions, key=lambda a: node.action_prob[a])
        
        child_env = node.env.clone()
        child_env.step(action)
        
        child_node = TreeNode(child_env)
        node.add_child(action, child_node)
        return child_node

    def _simulate(self, node):
        """Use network evaluation instead of random play"""
        if node.env._is_terminal():
            return (node.env.winner, 1)
            
        obs = node.env._get_observation()
        torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.policy(torch_x)
            
        # Return the value as the simulation result
        # Convert value from [-1, 1] to actual winner
        value = value.item()

        return (node.env.current_player, value)

    def _backpropagation(self, node, result):
        """Backpropagate the result"""
        current = node
        while current is not None:
            current.update(result)
            current = current.parent

    def _best_action(self, root: TreeNode):
        """Return the best action based on visit count"""
        if not root.children:
            return random.choice(root.env.get_valid_actions())
        return max(root.children.items(), key=lambda item: item[1].visits)[0]
    
    def select_action_with_temperature(self, temperature=1.0):
        if not self.root.children:
            # 如果没有子节点，随机选一个（虽然不太可能发生）
            return random.choice(self.root.env.get_valid_actions()), {}
    
        visit_counts = np.array([child.visits for child in self.root.children.values()])
        actions = list(self.root.children.keys())
    
        if temperature == 0:
            # 零温度等同于取最大值
            action_index = np.argmax(visit_counts)
            chosen_action = actions[action_index]
        else:
            # 根据温度调整概率分布
            visit_probs = visit_counts**(1 / temperature)
            visit_probs /= np.sum(visit_probs) # 归一化
            
            # 从分布中采样
            action_index = np.random.choice(len(actions), p=visit_probs)
            chosen_action = actions[action_index]
            
        # 同时，我们需要为训练准备 π (pi) 向量
        # π 是不加温度的、归一化的访问次数，代表了MCTS的“思考结果”
        pi_distribution = {a: node.visits for a, node in self.root.children.items()}
        total_visits = sum(pi_distribution.values())
        probs_for_training = [pi_distribution.get(i, 0) / total_visits for i in range(self.root.env.board_size**2)]
    
        return chosen_action, probs_for_training

    
    def _puct_value(self, child: TreeNode, action_prob=1.0):
        """Calculate PUCT value using the policy network"""
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.wins / child.visits
        exploration = self.puct * action_prob * math.sqrt(child.parent.visits) / (1 + child.visits)

        # if child.env.current_player != self.root.env.current_player:
        return -(exploitation) + exploration

#%%

if __name__ == '__main__':
    policy = ZeroPolicy(board_size=9)
    game = GomokuEnv(board_size=9)

    game.board[0, 1:4] = 1
    game.board[1, 1:4] = 2

    game.render()
    zero_player = WrongZeroMCTS(game, policy)
    # %%
    action = zero_player.run(300)
    print(f"Best action: {action}")
    # %%
    # zero_player.root.
    for child_action, child_node in zero_player.root.children.items():
        print(f"Action: {child_action}, Visits: {child_node.visits}, Wins: {child_node.wins}, Action Prob: {child_node.action_prob}")
    # %%

    game.get_valid_actions()
    # %%

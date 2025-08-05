#%%
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
    

#%% 
class TreeNode:
    def __init__(self, env: GomokuEnv):
        self.env = env
        self.visits = 0
        self.wins = 0
        self.children = {}
        self.parent = None
        self.action_prob = None

    def add_child(self, action, child_node):
        child_node.parent = self
        self.children[action] = child_node

    def update(self, result):
        self.visits += 1

        player = result[0]
        if player == self.env.current_player:
            self.wins += result[1]  # result[1] is the value from the simulation
        elif player == 3 - self.env.current_player:  # Assuming 1 and 2 are players
            self.wins -= result[1]
    
    def is_fully_expanded(self):
        """Check if the node is fully expanded"""
        valid_actions = self.env.get_valid_actions()
        return len(self.children) == len(valid_actions)

class ZeroTreeNode:
    def __init__(self, env: GomokuEnv, parent=None): # 添加 parent 初始化
        self.env = env
        self.visits = 0
        self.value_sum = 0  # 使用 value_sum 代替 wins，更通用
        self.children = {}
        self.parent = parent
        self.action_prob = None

    def add_child(self, action, child_node):
        self.children[action] = child_node

    # update 方法接收一个从子节点视角看的 value
    def update(self, value):
        self.visits += 1
        self.value_sum += value
    
    # Q-Value，即 exploitation 项
    @property
    def q_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def is_fully_expanded(self):
        """Check if the node is fully expanded"""
        valid_actions = self.env.get_valid_actions()
        return len(self.children) == len(valid_actions)

class MCTS:
    def __init__(self, env,strategy=RandomStrategy(), c=1.41, puct = 5):
        self.strategy = strategy
        self.c = c  # Exploration constant for UCT
        self.puct = puct
        self.root = TreeNode(env.clone())

    def run(self, iterations):
        """Run MCTS and return the best action"""
        for _ in range(iterations):
            node = self._select(self.root)
            
            if not node.env._is_terminal() and not node.is_fully_expanded():
                node = self._expansion(node)
            
            result = self._simulate(node)
            self._backpropagation(node, result)
        
        return self._best_action(self.root)
    
    def _select(self, node: TreeNode):
        """Select node using UCT"""
        current = node
        while not current.env._is_terminal() and current.is_fully_expanded():
            strategy_name = self.strategy.name()
            if strategy_name == 'random':
                current = max(current.children.values(), key=lambda child: self._uct_value(child))
            elif strategy_name == 'zero':
                current = max(current.children.values(), key=lambda child: self._puct_value(child))
            else:
                raise ValueError("unsupported strategy!")

        return current
    
    def _puct_value(self, child):
        pass

    def _expansion(self, node):
        """Expand the node"""
        valid_actions = node.env.get_valid_actions()
        expanded_actions = set(node.children.keys())
        available_actions = [a for a in valid_actions if a not in expanded_actions]
        
        if not available_actions:
            return node
            
        action = self.strategy.Choose(available_actions)
        child_env = node.env.clone()
        child_env.step(action)
        
        child_node = TreeNode(child_env)
        node.add_child(action, child_node)
        return child_node

    def _simulate(self, node):
        """Simulate a random game"""
        env = node.env.clone()
        while not env._is_terminal():
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                break
            action = self.strategy.Choose(valid_actions)
            env.step(action)
        return (env.winner, 1)

    def _backpropagation(self, node, result):
        """Backpropagate the result"""
        current = node
        while current is not None:
            current.update(result)
            current = current.parent
    
    def _best_action(self, root: TreeNode):
        """Return the best action based on visit count"""
        if not root.children:
            return self.strategy.Choose(root.env.get_valid_actions())
        return max(root.children.items(), key=lambda item: item[1].visits)[0]
    
    def _uct_value(self, child: TreeNode):
        """Calculate UCT value"""
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.wins / child.visits
        exploration = self.c * math.sqrt(math.log(child.parent.visits) / child.visits)
        
        # expoloration = self.c * P_a(Ation) * (sqrt(all visition)  / 1 + N(s, a)
        
        # For two-player games, we need to negate for the opponent's perspective
        if child.env.current_player != self.root.env.current_player:
            return -(exploitation) + exploration
        else:
            return exploitation + exploration


class ZeroMCTS:
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
    
    def _compute_policy_and_value(self, node: ZeroTreeNode):
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
    
    # def _select(self, node: ZeroTreeNode):
    #     current = node
    #     while current.children:
    #         current = max(current.children.values(), key=lambda child: self._puct_value(child))
    #     return current
    
    # def _expand_and_evaluate(self, node: ZeroTreeNode):
    #     if node.env._is_terminal():
    #         winner = node.env.winner
    #         if winner == 0:
    #             return 0.0
    #         if winner == (3-node.env.current_player):
    #             return -1.0
    #         else:
    #             return 1.0

    #     policy_probs, value = self._compute_policy_and_value(node) 

    #     valid_actions = node.env.get_valid_actions()

    #     for action in valid_actions:
    #         if action not in node.children:
    #             child_env = node.env.clone()
    #             child_env.step(action)
    #             node.add_child(action, ZeroTreeNode(child_env, parent=node))

    #     return value
    
    # def _backpropagation(self, node: ZeroTreeNode, value: float):
    #     current = node
    #     while current is not None:
    #         current.update(value)
    #         value = -value
    #         current = current.parent
    
    # def _puct_value(self, child: ZeroTreeNode):
    #     q_value = -child.q_value
    #     prior_prob = child.parent.action_prob[child.env.last_action]
        
    #     exploration_term = self.puct * prior_prob * \
    #                        (math.sqrt(child.parent.visits) / (1 + child.visits))
        
    #     return q_value + exploration_term
    
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

    def _best_action(self, root: ZeroTreeNode):
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

    game.board[0, :4] = 1
    game.board[1, :4] = 2

    game.render()
    zero_player = ZeroMCTS(game, policy)
    # %%
    action = zero_player.run(3000)
    print(f"Best action: {action}")
    # %%
    # zero_player.root.
    for child_action, child_node in zero_player.root.children.items():
        print(f"Action: {child_action}, Visits: {child_node.visits}, Wins: {child_node.wins}, Action Prob: {child_node.action_prob}")
    # %%

    game.get_valid_actions()
    # %%

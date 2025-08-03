#%%
from gomoku.gomoku_env import GomokuEnv
from gomoku.policy import ZeroPolicy
import torch.nn.functional as F
import torch
import random
import math
from abc import ABC, abstractmethod

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
    def __init__(self, env: GomokuEnv, policy: ZeroPolicy, puct=5):
        # Policy is evaluation network.
        self.env = env
        self.policy = policy
        self.puct = puct 
        self.root = TreeNode(env.clone())
    
    def run(self, iterations):
        """Run MCTS with PUCT using the policy network"""
        for _ in range(iterations):
            node = self._select(self.root)
            
            if not node.env._is_terminal() and not node.is_fully_expanded():
                node = self._expansion(node)
            
            result = self._simulate(node)
            self._backpropagation(node, result)
        
        return self._best_action(self.root) 
    
    def _select(self, node: TreeNode):
        """Select node using PUCT"""
        current = node
        while not current.env._is_terminal() and current.is_fully_expanded():
            if current.action_prob is None:
                obs = current.env._get_observation()
                torch_x = torch.from_numpy(obs).unsqueeze(0).float()
                valid_actions_tensor = torch.tensor(current.env.get_valid_actions())
                
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
            torch_x = torch.from_numpy(obs).unsqueeze(0).float()
            valid_actions_tensor = torch.tensor(node.env.get_valid_actions())
            
            # Get policy and value from network
            with torch.no_grad():
                torch_x = torch_x.to(device=self.policy.device)
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
        torch_x = torch.from_numpy(obs).unsqueeze(0).float()
        
        with torch.no_grad():
            torch_x = torch_x.to(device=self.policy.device)
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
    
    def _puct_value(self, child: TreeNode, action_prob=1.0):
        """Calculate PUCT value using the policy network"""
        if child.visits == 0:
            return float('inf')
        
        exploitation = child.wins / child.visits
        exploration = self.puct * action_prob * math.sqrt(child.parent.visits) / (1 + child.visits)

        if child.env.current_player != self.root.env.current_player:
            return -(exploitation) + exploration
        
        return exploitation + exploration

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

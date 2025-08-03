# %%
import torch.nn as nn
import torch
import torch.nn.functional as F

class ZeroPolicy(nn.Module):
    def __init__(self, board_size, device='cpu'):
        super(ZeroPolicy, self).__init__()
        self.board_size = board_size
        self.channel_size = 3 # 1. board state, 2. player turn, 3. Last action
        self.device = device
        
        # Shared trunk - simple conv layers as in AlphaZero
        self.conv1 = nn.Conv2d(self.channel_size, 32, kernel_size=3, padding=1, device=self.device)
        self.bn1 = nn.BatchNorm2d(32, device=self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, device=self.device)
        self.bn2 = nn.BatchNorm2d(64, device=self.device)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 2, kernel_size=1, device=self.device)
        self.policy_bn = nn.BatchNorm2d(2, device=self.device)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size, device=self.device)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, kernel_size=1, device=self.device)
        self.value_bn = nn.BatchNorm2d(1, device=self.device)
        self.value_fc1 = nn.Linear(board_size * board_size, 64, device=self.device)
        self.value_fc2 = nn.Linear(64, 1, device=self.device)

    def forward(self, x: torch.Tensor):
        # Input: (batch_size, channel_size * board_size * board_size)
        batch_size = x.size(0)
        
        # Reshape to (batch_size, channel_size, board_size, board_size)
        x = x.view(batch_size, self.channel_size, self.board_size, self.board_size)
        
        # Shared trunk
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(batch_size, -1)
        policy = self.policy_fc(policy)
        # policy = F.log_softmax(policy, dim=1)
        # policy = F.softmax(policy, dim=1)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(batch_size, -1)
        value = self.value_fc1(value)
        value = F.relu(value)
        value = self.value_fc2(value)
        value = F.tanh(value)
        
        return policy, value

    def get_action_prob(self, x: torch.Tensor, valid_actions=None):
        """Get action probabilities from the policy head."""
        logits = self.forward(x)['policy']
        mask = torch.ones_like(x)
        if valid_actions is not None:
            # mask
            mask = torch.index_fill(torch.zeros_like(logits), -1, valid_actions, 1)
            # sofxmax -> prob 
        prob = F.softmax(mask * logits, dim=-1)
        return prob

# %%
if __name__ == "__main__":
    # Example usage
    policy = ZeroPolicy(15)  # Example instantiation for a 15x15 board

    policy(torch.randn(1, 3 * 15 * 15))  # Example forward pass with random
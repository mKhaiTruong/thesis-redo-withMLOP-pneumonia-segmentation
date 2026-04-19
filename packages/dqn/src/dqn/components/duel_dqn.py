import torch.nn as nn

# Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super().__init__()

        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, x):
        shared = self.shared(x)
        value  = self.value_stream(shared)
        advantage = self.advantage_stream(shared)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
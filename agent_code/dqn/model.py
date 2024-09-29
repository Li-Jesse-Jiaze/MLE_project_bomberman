import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        
        # Feature extraction layers with Layer Normalization
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        
        # Value stream
        self.value_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        
        # Compute value and advantage
        value = self.value_layer(features)
        advantage = self.advantage_layer(features)
        
        # Combine value and advantage to get Q-values
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

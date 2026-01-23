"""PyTorch policy for Flappy Bird environment.

Optimized for training speed:
- Smaller default size (32 vs 64)
- 2 hidden layers with Tanh (needed for learning)
- No LSTM (feedforward only)
"""

import numpy as np
import torch
import torch.nn as nn

import pufferlib
import pufferlib.pytorch


class Policy(nn.Module):
    """Compact MLP policy for Flappy Bird.
    
    2-layer network balancing speed and learning capacity.
    Input: 8-dimensional observation
    Output: 2 action logits + 1 value estimate
    """
    
    def __init__(self, env, hidden_size=64, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        
        num_obs = np.prod(env.single_observation_space.shape)
        num_actions = env.single_action_space.n
        
        # 2-layer encoder (needed for learning, Tanh works better for RL)
        self.encoder = nn.Sequential(
            pufferlib.pytorch.layer_init(nn.Linear(num_obs, hidden_size)),
            nn.Tanh(),
            pufferlib.pytorch.layer_init(nn.Linear(hidden_size, hidden_size)),
            nn.Tanh(),
        )
        
        # Actor head (action logits)
        self.actor = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, num_actions), std=0.01
        )
        
        # Critic head (value function)
        self.critic = pufferlib.pytorch.layer_init(
            nn.Linear(hidden_size, 1), std=1.0
        )
        
    def forward(self, observations, state=None):
        """Forward pass returning action logits and value."""
        hidden = self.encode_observations(observations)
        actions, value = self.decode_actions(hidden)
        return actions, value
        
    def encode_observations(self, observations, state=None):
        """Encode observations into hidden representation."""
        batch_size = observations.shape[0]
        x = observations.view(batch_size, -1).float()
        return self.encoder(x)
        
    def decode_actions(self, hidden):
        """Decode hidden state to action logits and value."""
        action_logits = self.actor(hidden)
        value = self.critic(hidden)
        return action_logits, value

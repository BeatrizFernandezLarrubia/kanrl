import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy_MLP(nn.Module):
    def __init__(self, env, device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(np.array(env.observation_space.shape[0]).prod(), 256, device=self.device)
        self.fc2 = nn.Linear(256, 256, device=self.device)
        self.fc_mu = nn.Linear(256, np.prod(env.action_space.shape[0]), device=self.device)
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32).to(self.device)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias

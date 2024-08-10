import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Policy_MLP(nn.Module):
    def __init__(self, env, device, dimension_wrapper_number=0):
        super().__init__()
        self.device = device
        self.env = env
        self.node_count = 256
        self.fc1 = nn.Linear(np.array(env.observation_space.shape[0] + dimension_wrapper_number).prod(), self.node_count, device=self.device)
        self.fc2 = nn.Linear(self.node_count, self.node_count, device=self.device)
        if env.action_space.dtype == int:
            self.fc_mu = nn.Linear(self.node_count, np.prod(env.action_space.n), device=self.device)
        else:
            self.fc_mu = nn.Linear(self.node_count, np.prod(env.action_space.shape[0]), device=self.device)
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
        if self.env.action_space.dtype == int:
            x = F.softmax(self.fc_mu(x))
            return x
        else:
            x = torch.tanh(self.fc_mu(x))
            return x * self.action_scale + self.action_bias

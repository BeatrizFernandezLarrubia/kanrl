import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, observation_dim, action_dim):
        self.capacity = capacity
        self.observations = torch.zeros(capacity, observation_dim)
        # Update actions to accommodate continuous values and multiple dimensions
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32)
        self.next_observations = torch.zeros(capacity, observation_dim)
        self.rewards = torch.zeros(capacity, 1)
        self.terminations = torch.zeros(capacity, 1, dtype=torch.int)
        self.cursor = 0

    def add(self, observation, action, next_observation, reward, termination):
        index = self.cursor % self.capacity
        self.observations[index] = observation
        self.actions[index] = action
        self.next_observations[index] = next_observation
        self.rewards[index] = reward
        self.terminations[index] = termination

        self.cursor += 1

    def sample(self, batch_size):
        idx = np.random.choice(min(self.cursor, self.capacity), batch_size)
        return (
            self.observations[idx],
            self.actions[idx],
            self.next_observations[idx],
            self.rewards[idx],
            self.terminations[idx],
        )

    def __len__(self):
        return min(self.cursor, self.capacity)

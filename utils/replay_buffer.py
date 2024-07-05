import random

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha

    def push(self, *args):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = args
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        priorities = priorities ** self.alpha
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = list(zip(*samples))
        states = np.array(batch[0])
        actions = np.array(batch[1])
        rewards = np.array(batch[2])
        next_states = np.array(batch[3])
        dones = np.array(batch[4])

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

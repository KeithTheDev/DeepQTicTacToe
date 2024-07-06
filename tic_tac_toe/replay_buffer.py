import os
import random

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.priorities = []
        self.position = 0
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            self.priorities.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities[:self.position])
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

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

    def save(self, path):
        data = {
            'memory': self.memory,
            'priorities': self.priorities,
            'position': self.position
        }
        with open(path, 'wb') as f:
            np.save(f, data)

    def load(self, path):
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = np.load(f, allow_pickle=True).item()
                self.memory = data['memory']
                self.priorities = data['priorities']
                self.position = data['position']
        else:
            print(f"No replay buffer found at {path}. Starting with an empty buffer.")

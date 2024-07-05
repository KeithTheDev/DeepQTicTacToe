import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, architecture='default'):
        super(DQN, self).__init__()
        if architecture == 'default':
            self.fc1 = nn.Linear(9, 128)
            self.fc2 = nn.Linear(128, 128)
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, 9)
        elif architecture == 'small':
            self.fc1 = nn.Linear(9, 64)
            self.fc2 = nn.Linear(64, 64)
            self.value_stream = nn.Linear(64, 1)
            self.advantage_stream = nn.Linear(64, 9)
        elif architecture == 'large':
            self.fc1 = nn.Linear(9, 256)
            self.fc2 = nn.Linear(256, 256)
            self.value_stream = nn.Linear(256, 1)
            self.advantage_stream = nn.Linear(256, 9)
        else:
            raise ValueError("Unknown architecture type")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())

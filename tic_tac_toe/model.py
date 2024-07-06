import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, architecture='default'):
        super(DQN, self).__init__()
        if architecture == 'default':
            self.fc1 = nn.Linear(9, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, 128)
            self.fc5 = nn.Linear(128, 128)
            self.fc6 = nn.Linear(128, 128)
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, 9)
        elif architecture == 'small':
            self.fc1 = nn.Linear(9, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 64)
            self.fc4 = nn.Linear(64, 64)
            self.fc5 = nn.Linear(64, 64)
            self.fc6 = nn.Linear(64, 64)
            self.value_stream = nn.Linear(64, 1)
            self.advantage_stream = nn.Linear(64, 9)
        elif architecture == 'large':
            self.fc1 = nn.Linear(9, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 256)
            self.fc4 = nn.Linear(256, 256)
            self.fc5 = nn.Linear(256, 256)
            self.fc6 = nn.Linear(256, 256)
            self.value_stream = nn.Linear(256, 1)
            self.advantage_stream = nn.Linear(256, 9)
        elif architecture == 'extra_small':
            self.fc1 = nn.Linear(9, 32)
            self.fc2 = nn.Linear(32, 32)
            self.fc3 = nn.Linear(32, 32)
            self.fc4 = nn.Linear(32, 32)
            self.fc5 = nn.Linear(32, 32)
            self.fc6 = nn.Linear(32, 32)
            self.value_stream = nn.Linear(32, 1)
            self.advantage_stream = nn.Linear(32, 9)
        elif architecture == 'medium':
            self.fc1 = nn.Linear(9, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, 128)
            self.fc5 = nn.Linear(128, 128)
            self.fc6 = nn.Linear(128, 128)
            self.fc7 = nn.Linear(128, 128)  # Additional layer
            self.fc8 = nn.Linear(128, 128)  # Additional layer
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, 9)
        elif architecture == 'extra_large':
            self.fc1 = nn.Linear(9, 512)
            self.fc2 = nn.Linear(512, 512)
            self.fc3 = nn.Linear(512, 512)
            self.fc4 = nn.Linear(512, 512)
            self.fc5 = nn.Linear(512, 512)
            self.fc6 = nn.Linear(512, 512)
            self.value_stream = nn.Linear(512, 1)
            self.advantage_stream = nn.Linear(512, 9)
        elif architecture == 'deep':
            self.fc1 = nn.Linear(9, 128)
            self.fc2 = nn.Linear(128, 128)
            self.fc3 = nn.Linear(128, 128)
            self.fc4 = nn.Linear(128, 128)
            self.fc5 = nn.Linear(128, 128)
            self.fc6 = nn.Linear(128, 128)
            self.fc7 = nn.Linear(128, 128)
            self.fc8 = nn.Linear(128, 128)
            self.fc9 = nn.Linear(128, 128)
            self.fc10 = nn.Linear(128, 128)
            self.value_stream = nn.Linear(128, 1)
            self.advantage_stream = nn.Linear(128, 9)
        else:
            raise ValueError("Unknown architecture type")

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        x = torch.relu(self.fc6(x))
        if hasattr(self, 'fc7'):
            x = torch.relu(self.fc7(x))
        if hasattr(self, 'fc8'):
            x = torch.relu(self.fc8(x))
        if hasattr(self, 'fc9'):
            x = torch.relu(self.fc9(x))
        if hasattr(self, 'fc10'):
            x = torch.relu(self.fc10(x))
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + (advantage - advantage.mean())

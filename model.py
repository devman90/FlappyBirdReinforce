import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, game="pixelcopter"):
        super(DQN, self).__init__()
        self.number_of_actions = 2

        if game == "flappy":
            # 1, 4, 84, 84 / 1, 4, 48, 48
            self.conv1 = nn.Conv2d(4, 32, kernel_size = 8, stride = 4)
            # 1, 32, 20, 20 / 1, 32, 11, 11
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            # 1, 64, 9, 9 / 1, 64, 4, 4
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            # 1, 64, 7, 7 / 1, 64, 2, 2
            self.fc4 = nn.Linear(7 * 7 * 64, 512)
            self.fc5 = nn.Linear(512, self.number_of_actions)
        elif game == "pixelcopter":
            # 1, 4, 84, 84 / 1, 4, 48, 48
            self.conv1 = nn.Conv2d(4, 32, kernel_size = 8, stride = 4)
            # 1, 32, 20, 20 / 1, 32, 11, 11
            self.conv2 = nn.Conv2d(32, 64, 4, 2)
            # 1, 64, 9, 9 / 1, 64, 4, 4
            self.conv3 = nn.Conv2d(64, 64, 3, 1)
            # 1, 64, 7, 7 / 1, 64, 2, 2
            self.fc4 = nn.Linear(2 * 2 * 64, 512)
            self.fc5 = nn.Linear(512, self.number_of_actions)


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)

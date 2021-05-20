#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


class Policy(nn.Module):
    def __init__(
        self,
        dim_action,
        max_action=None,
        activation=nn.LeakyReLU
    ):
        super(Policy, self).__init__()
        self.dim_action = dim_action
        self.max_action = max_action

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(32)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(32)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, self.dim_action)

        self.apply(init_weight)

    def forward(self, x):
        x = x.to(device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        action = self.head(x.view(x.size(0), -1))
        return action * self.max_action

    def get_action_log_prob(self, states):
        action = self.forward(states)
        return action, None


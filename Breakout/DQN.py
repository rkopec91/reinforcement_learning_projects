import os
import random
import gym
import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
import torchvision as tv

TRAIN = True
ENVIRONMENT = 'BreakoutDeterministic-v4'



class DQN(nn.Module):
    def __init__(self, n_actions, hidden = 1024, learning_rate=1e-5, frame_height=84, frame_width=84, history_length=4):
        super(DQN, self).__init__()
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.history_length = history_length


        self.conv1 = nn.Conv2d(4,32,kernel_size=8,stride=4, bias=False)
        self.conv2 = nn.Conv2d(32,64, kernel_size=4, stride=2, bias=False)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1, bias=False)
        self.conv4 = nn.Conv2d(64,hidden,kernel_size=7,stride=1, bias=False)

        self.valuestream, self.advantagestream = torch.split(self.conv4, 2, 3)
        self.valuestream = torch.flatten(self.valuestream)
        self.advantagestream = torch.flatten(self.advantagestream)

        self.advantage = nn.Linear(self.advantagestream, self.n_actions, False)

        self.value = nn.Linear(self.valuestream, 1, False)

        self.q_values = self.value + self.advantage.sub(torch.mean(self.advantage,dim=1, keepdim=True))
        self.best_action = torch.argmax(self.q_values, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return(x)

dqn = DQN(4)
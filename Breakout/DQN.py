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


        self.func1_adv = nn.Linear(in_features=7*7*64, out_features=512)
        self.func1_val = nn.Linear(in_features=7*7*64, out_features=512)

        self.func2_adv = nn.Linear(in_features=512, out_features=n_actions)
        self.func2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.func1_adv(x))
        val = self.relu(self.func1_val(x))

        adv = self.func2_adv(adv)
        val = self.func2_val(val).expand(x.size(0), self.n_actions)
        
        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.n_actions)
        return x

dqn = DQN(4)
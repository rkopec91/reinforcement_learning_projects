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
        self.n_actions = n_actions
        self.hidden = hidden
        self.learning_rate = learning_rate
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.history_length = history_length


        self.conv1 = nn.Conv2d(4,32,kernel_size=8,stride=4, bias=False)
        self.conv2 = nn.Conv2d(self.conv1,64, kernel_size=4, strides=2, bias=False)
        self.conv3 = nn.Conv2d(self.conv2,64, kernel_size=3, strides=1, bias=False)
        self.conv4 = nn.Conv2d(self.conv3,hidden,kernel_size=7,strides=1, bias=False)

        self.valuestream, self.advantagestream = torch.split(self.conv4, 2, 3)
        self.valuestream = torch.fla
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return(x)

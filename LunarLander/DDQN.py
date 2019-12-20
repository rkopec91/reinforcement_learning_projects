import os
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

class DDQN(nn.Module):
    def __init__(self, alpha, actions, name, input):
        super(DDQN, self).__init__()

        self.fc1 = nn.Linear(*input, 128)
        self.fc2 = nn.Linear(128,128)
        self.v = nn.Linear(128, 1)
        self.a = nn.Linear(128, actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.loss = nn.MSELoss()
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device('cpu')

        self.name = name
        self.to(self.device)
        self.checkpoint_dir = './checkpoints/'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        layer2 = F.relu(self.fc2(layer1))
        v = self.v(layer2)
        a = self.a(layer2)

        return v, a

    def save(self):
        print('saving to {}...'.format(self.name))
        torch.save(self.state_dict(), self.checkpoint_file)

    def load(self):
        print('loading {}...'.format(self.name))
        self.load_state_dict(torch.load(self.checkpoint_file))
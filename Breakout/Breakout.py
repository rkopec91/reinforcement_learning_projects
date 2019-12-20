import os
import random
import gym
import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
import torchvision as tv
from ImageProcessor import ImageProcessor


class Game:
    def __init__(self, env_name='BreakoutDeterministic-v4', steps=10, history_length=4):
        '''
        env_name: name of the game the agent is playing from gym.
        BreakoutDeterministic-v4 or BreakoutDeterministic-v3

        steps:  the number of steps the agent sill take

        history_length:  Length of the history
        '''
        self.env = gym.make(env_name)
        self.state = None
        self.lives = 0
        self.steps = steps
        self.history_length = history_length
        self.process_frame = ImageProcessor()

    def reset_game(self, evaluation=False):
        frame = self.env.reset()
        self.lives = 0
        terminal_life = True

        if evaluation:
            for _ in range(random.randint(1,self.steps)):
                frame, _, _, _ = self.env.step(1)

        processed_frame = self.process_frame(frame)
        self.state = np.repeat(processed_frame, self.agent_history_length, axis=2)

        return terminal_life

    def step(self, action):

        new_frame, reward, terminal, info = self.env.step(action)

        if info['ale.lives'] < self.last_lives:
            terminal_life_lost = True
        else:
            terminal_life_lost = terminal
        self.last_lives = info['ale.lives']

        processed_new_frame = self.process_frame(new_frame)
        new_state = np.append(self.state[:, :, 1:], processed_new_frame, axis=2)
        self.state = new_state

        return processed_new_frame, reward, terminal, terminal_life_lost, new_frame

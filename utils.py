import collections
import cv2
import numpy as np
import gym

class RepeatActionAndMaxFrame(gym.wrapper):
    def __init__(self,env=None, repeat=4):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        slef.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2,self.shape))
    
    def step(self, action):
        t_reward = 0.0
        done = False
        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = obs
            if done:
                break
        
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, t_reward, done, info
    
    def reset(self):
        obs = self.env.reset()
        self.frame_buffer = np.zeros_like((2,self.shape))
        self.frame_buffer[0] = obs
        
        return obs
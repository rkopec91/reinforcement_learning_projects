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
    
class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        #Update shape to have channel first
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, 
                                                heigh=1.0, 
                                                shape=self.shape, 
                                                dtype=np.float32)
    
    def observation(self, observation):
        new_frame = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_observation = np.array(resized_screen, dtype=np.int8).reshape(self.shape) / 255.0
        return new_observation
        
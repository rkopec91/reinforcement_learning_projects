from Replay import Replay
from DDQN import DDQN
import numpy as np
import torch

class Agent:
    def __init__(self, gamma, epsilon, alpha, actions, input_dimension, memory_size, batch_size, minimum_eps=0.01, eps_decay=5e-7, replace=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.minimum_eps = minimum_eps
        self.eps_decay = eps_decay
        self.actions = [i for i in range(actions)]
        self.step = 0
        self.replace = replace
        self.batch_size = batch_size

        self.memory = Replay(memory_size, input_dimension, actions)
        self.q_eval = DDQN(alpha, actions, input=input_dimension, name="evaluation")
        self.q_next = DDQN(alpha, actions, input=input_dimension, name="evaluation_next")

    def record_transition(self, state, action, reward, next_state, done):
        self.memory.record_transition(state, action, reward, next_state, done)

    def choose_action(self, obs):
        if np.random.random() > self.epsilon:
            obs = obs[np.newaxis, :]
            state = torch.tensor(obs).to(self.q_eval.device)
            value, advantage = self.q_eval.forward(state)
            action = torch.argmax(advantage).item()

        else:
            action = np.random.choice(self.actions)

        return action

    def replace_network(self):
        if self.replace is not None and self.step % self.replace == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def reduce_epsilon(self):
        self.epsilon = self.epsilon - self.eps_decay if self.epsilon > self.minimum_eps else self.minimum_eps

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        #zero out gradients
        self.q_eval.optimizer.zero_grad()
        self.replace_network()

        state, action, reward, new_state, done = self.memory.sample_buff(self.batch_size)

        state = torch.tensor(state).to(self.q_eval.device)
        new_state = torch.tensor(new_state).to(self.q_eval.device)
        action = torch.tensor(action).to(self.q_eval.device)
        reward = torch.tensor(reward).to(self.q_eval.device)
        done = torch.tensor(done).to(self.q_eval.device)
        
        Vs, As = self.q_eval.forward(state)
        new_Vs, new_As = self.q_eval.forward(new_state)
        
        prediction = torch.add(Vs, (As - As.mean(dim=1, keepdim=True))).gather(1, action.unsqueeze(-1)).squeeze(-1)
        prediction_new = torch.add(new_Vs, (new_As - new_As.mean(dim=1, keepdim=True)))
        
        target = reward + self.gamma*torch.max(prediction_new, dim=1)[0].detach()
        target[done] = 0.0
        
        loss = self.q_eval.loss(target, prediction).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.step += 1
        
        self.reduce_epsilon()
        
    def save(self):
        self.q_eval.save()
        self.q_next.save()
        
    def load(self):
        self.q_eval.load()
        self.q_eval.load()
import numpy as np


class Replay:
    """
        Create the replay buffer
    """
    def __init__(self, max_size, input_shape, actions):
        """
            Initialize the memory of the replay buffer
            max_size: maximum size of the memory
            input_shape: the shape of the input
            actions: number of acitons
        """
        self.size = max_size
        self.memory_counter = 0
        self.state_mem = np.zeros((self.size, *input_shape), dtype=np.float32)
        self.new_state_mem = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_mem = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.mem_size, dtype=np.float32)
        self.term_mem = np.zeros(self.mem_size, dtype=np.uint8)

    def record_transition(self, state, action, reward, new_state, done):
        """
            Save the current transition to memory
            state: current state
            action: action taken
            reward: reward received
            new_state: the new state
            done: whether it is at a terminal state or not
        """
        i = self.memory_counter % self.size
        self.state_mem[i] = state
        self.action_mem[i] = action
        self.reward_mem[i] = reward
        self.new_state_mem[i] = new_state
        self.term_mem[i] = done
        self.memory_counter += 1

    def sample_buff(self, batch_size):
        """
            samples from the memory
            batch_size: the amount of samples retrieved from memory.
        """
        batch = np.random.choice(min(self.memory_counter, self.size), batch_size, replace=False)

        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        new_states = self.new_state_mem[batch]
        dones = self.term_mem[batch]

        return states, actions, rewards, new_states, dones

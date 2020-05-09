from collections import deque
from numpy import random


class ReplayMemory:

    def __init__(self, size=20000):
        # replay memory is essentially a circular buffer
        # in python this is most similar to a deque initialized with a max_len argument
        self.memories = deque([], maxlen=size)
        self.mem_size = size

    def remember(self, experience):
        # experience is a 4-tuple vector
        #   current state
        #   new state
        #   reward
        #   game over
        self.memories.append(experience)

    def sample(self, size=512):
        # this needs to return a random sample of the experiences in
        # memory for when we want to train
        samples = []
        if len(self.memories) == self.mem_size:
            for _ in range(size):
                samples.append(self.memories[random.randint(0, self.mem_size)])
        return samples

from collections import deque
import random
from dqn import Experience

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, experience: Experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
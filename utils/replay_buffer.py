import random
from collections import deque
import torch

class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a).unsqueeze(1),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1),
            torch.stack(s2),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1)
        )


    def __len__(self):
        return len(self.buffer)

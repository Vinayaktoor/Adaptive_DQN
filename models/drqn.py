import torch
import torch.nn as nn

class DRQN(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden=128):
        super().__init__()
        self.rnn = nn.GRU(obs_dim, hidden, batch_first=True)
        self.q = nn.Linear(hidden, action_dim)

    def forward(self, x):
        _, h = self.rnn(x)
        return self.q(h.squeeze(0))

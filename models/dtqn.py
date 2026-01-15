import torch
import torch.nn as nn

class DTQN(nn.Module):
    def __init__(self, obs_dim, action_dim, d_model=128, layers=4):
        super().__init__()
        self.embed = nn.Linear(obs_dim, d_model)
        encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=4, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder, layers)
        self.q = nn.Linear(d_model, action_dim)

    def forward(self, x):
        x = self.embed(x)
        x = self.encoder(x)
        return self.q(x[:, -1])

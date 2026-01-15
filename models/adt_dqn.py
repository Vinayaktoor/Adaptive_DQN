import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        f = self.ff(x)
        return self.ln2(x + f)

class ADTDQN(nn.Module):
    def __init__(self, obs_dim, action_dim, d_model=128, L_max=6):
        super().__init__()
        self.L_max = L_max
        self.embed = nn.Linear(obs_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, 4) for _ in range(L_max)
        ])
        self.q_heads = nn.ModuleList([
            nn.Linear(d_model, action_dim) for _ in range(L_max)
        ])
        self.halt_heads = nn.ModuleList([
            nn.Linear(d_model, 1) for _ in range(L_max)
        ])

    def forward(self, x):
        x = self.embed(x)
        Qs, Ps = [], []

        for l in range(self.L_max):
            x = self.layers[l](x)
            h = x[:, -1]
            Qs.append(self.q_heads[l](h))
            Ps.append(torch.sigmoid(self.halt_heads[l](h)))

        return Qs, Ps

    def adaptive_depth(self, Ps, tau=0.6, min_depth=2):
        for i, p in enumerate(Ps):
            if i + 1 >= min_depth and p.mean() > tau:
                return i + 1
        return self.L_max

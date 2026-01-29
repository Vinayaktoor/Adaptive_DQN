import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        a, _ = self.attn(x, x, x)
        x = self.ln1(x + a)
        f = self.ff(x)
        return self.ln2(x + f)


class ADTDQN(nn.Module):
    def __init__(self, obs_dim, action_dim, d_model=128, max_depth=6):
        super().__init__()
        self.max_depth = max_depth

        self.embed = nn.Linear(obs_dim, d_model)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, heads=4)
            for _ in range(max_depth)
        ])

        self.q_heads = nn.ModuleList([
            nn.Linear(d_model, action_dim)
            for _ in range(max_depth)
        ])

    def forward(self, x):
        """
        x: (B, T, obs_dim)
        """
        x = self.embed(x)

        Qs = []
        uncertainties = []

        for l in range(self.max_depth):
            x = self.layers[l](x)
            h = x[:, -1]                    # last token
            Q = self.q_heads[l](h)          # (B, A)
            Qs.append(Q)

            # Uncertainty = variance across actions
            # U = torch.var(Q, dim=1, unbiased=False)  # (B,)
            U = (Q.max(dim=1)[0] - Q.min(dim=1)[0]) / (Q.abs().mean(dim=1) + 1e-6)
            uncertainties.append(U)

        return Qs, uncertainties
    def adaptive_depth(self, uncertainties, min_depth=2):
        """
        Choose depth with minimum uncertainty, subject to min_depth
        """
        u_vals = torch.stack([u.mean() for u in uncertainties])
        u_vals = u_vals / (u_vals + 1.0)  # normalization

        # ignore shallow depths
        u_vals[:min_depth-1] = float("inf")

        d = torch.argmin(u_vals).item() + 1 
        return d

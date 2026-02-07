import torch
import torch.nn.functional as F


def train_dqn(model, target, buffer, opt, gamma=0.99):
    s, a, r, s2, d = buffer.sample(32)

    # Ensure correct shapes
    a = a.unsqueeze(1)           # (B, 1)
    d = d.float().unsqueeze(1)   # (B, 1)
    r = r.unsqueeze(1)           # (B, 1)

    # Q(s,a)
    q = model(s).gather(1, a)

    with torch.no_grad():
        q_next = target(s2).max(1, keepdim=True)[0]
        y = r + gamma * (1 - d) * q_next

    loss = F.mse_loss(q, y)

    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item()


def train_adt(model, target, buffer, opt, gamma=0.99, lambda_comp=0.001):
    s, a, r, s2, d = buffer.sample(32)

    a = a.unsqueeze(1)           # (B, 1)
    d = d.float().unsqueeze(1)   # (B, 1)
    r = r.unsqueeze(1)           # (B, 1)

    # Forward pass (adaptive model)
    Qs, Ps = model(s)   # Qs: list[L][B, A]
    depth = model.adaptive_depth(Ps)

    with torch.no_grad():
        Q_target, _ = target(s2)
        q_next = Q_target[-1].max(1, keepdim=True)[0]
        y = r + gamma * (1 - d) * q_next

    # Bellman loss across depths
    loss = 0.0
    for l in range(model.max_depth):
        loss += F.mse_loss(Qs[l].gather(1, a), y)

    depth*=0.5 
    loss += lambda_comp * float(depth)


    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), depth

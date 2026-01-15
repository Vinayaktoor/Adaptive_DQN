import torch
import torch.nn.functional as F

def train_dqn(model, target, buffer, opt, gamma=0.99):
    s, a, r, s2, d = buffer.sample(32)
    q = model(s).gather(1, a)
    with torch.no_grad():
        y = r + gamma * (1-d) * target(s2).max(1, keepdim=True)[0]
    loss = F.mse_loss(q, y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()

def train_adt(model, target, buffer, opt, gamma=0.99, lambda_comp=0.001):
    s, a, r, s2, d = buffer.sample(32)
    Qs, Ps = model(s)
    depth = model.adaptive_depth(Ps)

    with torch.no_grad():
        Q_target, _ = target(s2)
        y = r + gamma * (1-d) * Q_target[-1].max(1, keepdim=True)[0]

    loss = 0
    for l in range(depth):
        loss += F.mse_loss(Qs[l].gather(1, a), y)

    loss += lambda_comp * depth
    opt.zero_grad()
    loss.backward()
    opt.step()

    return loss.item(), depth

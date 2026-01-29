# test_uncertainty.py

import gymnasium as gym
import minigrid
import torch

from models.adt_dqn import ADTDQN
from utils.preprocess import preprocess_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- config ----
ENV_NAME = "MiniGrid-MemoryS9-v0"
SEQ_LEN = 8
TAU = 0.6
# ----------------

env = gym.make(ENV_NAME)

act_dim = env.action_space.n
obs_dim = 7 * 7 * 3 + 1

model = ADTDQN(obs_dim, act_dim).to(device)
model.eval()

# ---- reset env ----
obs, _ = env.reset()
s = preprocess_obs(obs).to(device)

history = []

print("\nRunning uncertainty inspection...\n")

for step in range(20):
    history.append(s)
    history = history[-SEQ_LEN:]

    # pad history
    if len(history) < SEQ_LEN:
        pad = [torch.zeros(obs_dim, device=device) for _ in range(SEQ_LEN - len(history))]
        hist = pad + history
    else:
        hist = history

    x = torch.stack(hist).unsqueeze(0)  # (1, T, obs_dim)

    with torch.no_grad():
        Qs, uncertainties = model(x)

        # print uncertainties per depth
        u_vals = [round(u.mean().item(), 4) for u in uncertainties]
        d = model.adaptive_depth(uncertainties)

    print(f"Step {step:02d} | Uncertainty per depth: {u_vals} | Selected depth: {d}")

    # random action (we are not testing policy quality)
    a = torch.randint(act_dim, (1,)).item()
    obs2, _, terminated, truncated, _ = env.step(a)

    if terminated or truncated:
        print("\nEpisode ended early.\n")
        break

    s = preprocess_obs(obs2).to(device)

print("\nDone.\n")

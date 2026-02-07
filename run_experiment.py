import argparse
import json
import random
import numpy as np
import torch
import gymnasium as gym
import minigrid

from utils.preprocess import preprocess_obs
from utils.replay_buffer import ReplayBuffer
from utils.trainer import train_dqn, train_adt

from models.adt_dqn import ADTDQN
from models.dtqn import DTQN
from models.drqn import DRQN
from models.dqn import DQN

# ------------------ CONFIG ------------------
ENV_NAME = "MiniGrid-MemoryS9-v0"
EPISODES = 300
SEQ_LEN = 8
OBS_DIM = 7 * 7 * 3 + 1
LR = 1e-3
# --------------------------------------------


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pad_sequence(seq, seq_len, obs_dim, device):
    if len(seq) < seq_len:
        pad = [torch.zeros(obs_dim, device=device) for _ in range(seq_len - len(seq))]
        seq = pad + seq
    return seq


def make_model(name, obs_dim, act_dim):
    if name == "adt_dqn":
        return ADTDQN(obs_dim, act_dim)
    if name == "dtqn":
        return DTQN(obs_dim, act_dim)
    if name == "drqn":
        return DRQN(obs_dim, act_dim)
    if name == "dqn":
        return DQN(obs_dim, act_dim)
    raise ValueError("Unknown model")


def run(model_name, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    env = gym.make(ENV_NAME)
    act_dim = env.action_space.n

    model = make_model(model_name, OBS_DIM, act_dim).to(device)
    target = make_model(model_name, OBS_DIM, act_dim).to(device)
    target.load_state_dict(model.state_dict())

    buffer = ReplayBuffer()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    logs = {
        "episode": [],
        "reward": [],
        "avg_depth_ep": [],
        "success": []
    }

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=seed)
        s = preprocess_obs(obs).to(device)

        hist = []
        episode_depths = []
        ep_reward = 0
        done = False

        while not done:
            hist.append(s)
            hist = hist[-SEQ_LEN:]

            hist_pad = pad_sequence(hist, SEQ_LEN, OBS_DIM, device)
            x = torch.stack(hist_pad).unsqueeze(0)

            with torch.no_grad():
                if model_name == "adt_dqn":
                    Qs, uncertainties = model(x)
                    d = model.adaptive_depth(uncertainties)
                    a = Qs[d - 1].argmax().item()
                    episode_depths.append(d)
                else:
                    Q = model(x)
                    a = Q.argmax().item()

            obs2, r, term, trunc, _ = env.step(a)
            done = term or trunc
            ep_reward += r

            s2 = preprocess_obs(obs2).to(device)
            hist2 = hist + [s2]
            hist2 = hist2[-SEQ_LEN:]
            hist2_pad = pad_sequence(hist2, SEQ_LEN, OBS_DIM, device)
            x2 = torch.stack(hist2_pad)

            buffer.push(x.squeeze(0), a, r, x2, done)
            s = s2

            if len(buffer) > 50:
                if model_name == "adt_dqn":
                    train_adt(model, target, buffer, optimizer)
                else:
                    train_dqn(model, target, buffer, optimizer)

        target.load_state_dict(model.state_dict())

        logs["episode"].append(ep)
        logs["reward"].append(ep_reward)
        logs["success"].append(1 if ep_reward > 0 else 0)

        if episode_depths:
            logs["avg_depth_ep"].append(sum(episode_depths) / len(episode_depths))
        else:
            logs["avg_depth_ep"].append(None)

        print(f"[{model_name} | seed {seed}] Ep {ep:03d} | R={ep_reward}")

    with open(f"logs/{model_name}_seed{seed}.json", "w") as f:
        json.dump(logs, f)

    print(f"Saved logs/{model_name}_seed{seed}.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["adt_dqn", "dtqn", "drqn", "dqn"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run(args.model, args.seed)

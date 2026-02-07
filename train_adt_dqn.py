import gymnasium as gym
import minigrid
import json
import torch
from models import dqn, drqn, dtqn, adt_dqn
from utils.replay_buffer import ReplayBuffer
from utils.trainer import train_dqn, train_adt
from utils.preprocess import preprocess_obs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WARMUP_EPISODES = 0
seq_len = 8

# MiniGrid env
env = gym.make("MiniGrid-MemoryS9-v0")

act_dim = env.action_space.n
obs_dim = 7 * 7 * 3 + 1  # image + direction


def pad_sequence(seq, seq_len, obs_dim):
    if len(seq) < seq_len:
        pad = [torch.zeros(obs_dim, device=seq[0].device) for _ in range(seq_len - len(seq))]
        seq = pad + seq
    return seq


def run(model, target, train_fn,SEED, episodes=300):
    logs = {
    "episode": [],
    "reward": [],
    "avg_depth_ep": [],
    "success": []
    }

    model.to(device)
    target.to(device)

    buffer = ReplayBuffer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rewards = []
    depths = []        # depths from training updates
    all_depths = []    # depths from action selection (analysis)

    for ep in range(episodes):
        obs, _ = env.reset(seed=SEED)
        s = preprocess_obs(obs).to(device)

        hist = []
        episode_depths = []
        ep_r = 0
        done = False

        # annealed tau (much more stable)
        tau = max(0.3, 1.0 - ep / episodes)
        prev_d = model.max_depth
        while not done:
            hist.append(s)
            hist = hist[-seq_len:]

            hist_padded = pad_sequence(hist, seq_len, obs_dim)
            x = torch.stack(hist_padded).unsqueeze(0).to(device)

            with torch.no_grad():
                if train_fn == train_adt:
                    Qs, Ps = model(x)

                    if ep < WARMUP_EPISODES:
                        d = model.max_depth
                    else:
                        d_raw = model.adaptive_depth(Ps)
                        d = int(0.7 * prev_d + 0.3 * d_raw)
                        d = max(2, min(d, model.max_depth))
                        prev_d = d

                    a = Qs[d - 1].argmax(dim=-1).item()

                    episode_depths.append(d)
                    all_depths.append(d)
                    

                else:
                    a = model(x).argmax(dim=-1).item()

            obs2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_r += r

            s2 = preprocess_obs(obs2).to(device)

            hist2 = hist.copy()
            hist2.append(s2)
            hist2 = hist2[-seq_len:]

            hist2_padded = pad_sequence(hist2, seq_len, obs_dim)
            x2 = torch.stack(hist2_padded).to(device)

            buffer.push(
                x.squeeze(0),
                a,
                r,
                x2,
                done
            )

            s = s2

            if len(buffer) > 200:
                out = train_fn(model, target, buffer, opt)
                if train_fn == train_adt:
                    _, d_used = out
        
                    depths.append(d_used)
        logs["episode"].append(ep)
        logs["reward"].append(ep_r)
        logs["avg_depth_ep"].append(sum(episode_depths) / len(episode_depths))
        logs["success"].append(1 if ep_r > 0 else 0)
    
           
        rewards.append(ep_r)
        target.load_state_dict(model.state_dict())

        if depths:
            avg_depth = sum(depths) / len(depths)
            print(
                f"Episode {ep:03d} | "
                f"Reward: {ep_r:4.0f} | "
                f"Avg Depth (train): {avg_depth:.2f} | "
                f"Avg Depth (ep): {sum(episode_depths)/len(episode_depths):.2f}"
            )
        else:
            print(f"Episode {ep:03d} | Reward: {ep_r:4.0f}")
   
    with open("logs/adt_dqn_seed0.json", "w") as f:
        json.dump(logs, f)

    return rewards, depths, all_depths


if __name__ == "__main__":
    from models.adt_dqn import ADTDQN
    import random
    import numpy as np
    import torch

    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = ADTDQN(obs_dim, act_dim)
    target = ADTDQN(obs_dim, act_dim)
    target.load_state_dict(model.state_dict())

    rewards, depths, all_depths = run(model, target, train_adt,SEED)

    print("\nTraining finished")
    print("Average reward:", sum(rewards) / len(rewards))
    if depths:
        print("Average depth (train):", sum(depths) / len(depths))
    if all_depths:
        print("Average depth (policy):", sum(all_depths) / len(all_depths))

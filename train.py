import gymnasium as gym
import torch
from models import dqn, drqn, dtqn, adt_dqn
from utils.replay_buffer import ReplayBuffer
from utils.trainer import train_dqn, train_adt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WARMUP_EPISODES = 50

env = gym.make("CartPole-v1")
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n
seq_len = 8
def pad_sequence(seq, seq_len, obs_dim):
    if len(seq) < seq_len:
        pad = [torch.zeros(obs_dim) for _ in range(seq_len - len(seq))]
        seq = pad + seq
    return seq

def run(model, target, train_fn, episodes=200):
    model.to(device)
    target.to(device)

    buffer = ReplayBuffer()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    rewards, depths = [], []

    for ep in range(episodes):
        s, _ = env.reset()
        hist = []

        ep_r = 0
        done = False

        while not done:
            hist.append(torch.tensor(s, dtype=torch.float))
            hist = hist[-seq_len:]

            hist_padded = pad_sequence(hist, seq_len, obs_dim)
            x = torch.stack(hist_padded).unsqueeze(0)


            with torch.no_grad():
                if train_fn == train_adt:
                    Qs, Ps = model(x)
                    
                    if ep <WARMUP_EPISODES:
                        d = model.L_max
                    else:
                        d = model.adaptive_depth(Ps, tau=0.6)    
                    a = Qs[d - 1].argmax().item()
                else:
                    a = model(x).argmax().item()

            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            ep_r += r

            # build next history correctly
            hist2 = hist.copy()
            hist2.append(torch.tensor(s2, dtype=torch.float))
            hist2 = hist2[-seq_len:]

            hist2_padded = pad_sequence(hist2, seq_len, obs_dim)
            x2 = torch.stack(hist2_padded)

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
                    _, d = out
                    depths.append(d)

        rewards.append(ep_r)
        target.load_state_dict(model.state_dict())

        print(
            f"Episode {ep:03d} | "
            f"Reward: {ep_r:4.0f} | "
            f"Avg Depth: {sum(depths)/len(depths):.2f}" if depths else ""
        )

    return rewards, depths

if __name__ == "__main__":
    from models.adt_dqn import ADTDQN

    model = ADTDQN(obs_dim, act_dim)
    target = ADTDQN(obs_dim, act_dim)
    target.load_state_dict(model.state_dict())

    rewards, depths = run(model, target, train_adt)

    print("Training finished")
    print("Average reward:", sum(rewards) / len(rewards))
    if depths:
        print("Average depth:", sum(depths) / len(depths))

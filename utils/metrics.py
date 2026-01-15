import numpy as np
import time

class MetricsTracker:
    def __init__(self):
        self.episode_rewards = []
        self.depths = []
        self.losses = []
        self.times = []

    def log_episode(self, reward, depth=None, duration=None):
        self.episode_rewards.append(reward)
        if depth is not None:
            self.depths.append(depth)
        if duration is not None:
            self.times.append(duration)

    def log_loss(self, loss):
        self.losses.append(loss)

    def summary(self):
        return {
            "avg_return": float(np.mean(self.episode_rewards)),
            "std_return": float(np.std(self.episode_rewards)),
            "avg_depth": float(np.mean(self.depths)) if len(self.depths) > 0 else None,
            "avg_loss": float(np.mean(self.losses)) if len(self.losses) > 0 else None,
            "avg_episode_time": float(np.mean(self.times)) if len(self.times) > 0 else None,
        }

    def pretty_print(self, name="Model"):
        s = self.summary()
        print(f"\n=== {name} Metrics ===")
        print(f"Average Return     : {s['avg_return']:.2f}")
        print(f"Return Std         : {s['std_return']:.2f}")
        if s["avg_depth"] is not None:
            print(f"Average Depth Used : {s['avg_depth']:.2f}")
        if s["avg_loss"] is not None:
            print(f"Average Loss       : {s['avg_loss']:.4f}")
        if s["avg_episode_time"] is not None:
            print(f"Avg Episode Time   : {s['avg_episode_time']:.3f}s")

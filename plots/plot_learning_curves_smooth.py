import json
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "logs"
MODELS = ["dqn", "drqn", "dtqn", "adt_dqn"]
SEED = 0
WINDOW = 20  # smoothing window


def moving_average(x, window):
    x = np.array(x)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


plt.figure(figsize=(8, 5))

for model in MODELS:
    path = f"{LOG_DIR}/{model}_seed{SEED}.json"
    with open(path) as f:
        logs = json.load(f)

    success = logs["success"]
    smoothed = moving_average(success, WINDOW)

    plt.plot(
        range(len(smoothed)),
        smoothed,
        label=model.upper()
    )

plt.xlabel("Episode")
plt.ylabel(f"Success Rate (Moving Avg, window={WINDOW})")
plt.title("MiniGrid-MemoryS9: Smoothed Learning Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

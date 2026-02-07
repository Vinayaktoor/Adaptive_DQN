import json
import matplotlib.pyplot as plt

with open("logs/adt_dqn_seed0.json") as f:
    logs = json.load(f)

depths = [d for d in logs["avg_depth_ep"] if d is not None]

plt.hist(depths, bins=10)
plt.xlabel("Depth")
plt.ylabel("Frequency")
plt.title("ADT-DQN Depth Usage")
plt.show()

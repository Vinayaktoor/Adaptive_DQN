import json
import matplotlib.pyplot as plt

with open("logs/adt_dqn_seed0.json") as f:
    logs = json.load(f)

depths = logs["avg_depth_ep"]

plt.plot(depths)
plt.xlabel("Episode")
plt.ylabel("Avg Depth per Episode")
plt.title("ADT-DQN Adaptive Depth")
plt.grid()
plt.show()

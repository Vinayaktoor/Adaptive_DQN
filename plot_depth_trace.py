import json
import matplotlib.pyplot as plt

with open("logs/adt_dqn_seed0.json") as f:
    logs = json.load(f)

depths = logs["avg_depth_ep"]

plt.plot(depths)
plt.xlabel("Episode")
plt.ylabel("Average Depth")
plt.title("ADT-DQN: Avg Depth per Episode")
plt.show()

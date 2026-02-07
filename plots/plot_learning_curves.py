import json
import matplotlib.pyplot as plt

def load(model, seed=0):
    with open(f"logs/{model}_seed{seed}.json") as f:
        return json.load(f)["reward"]

models = ["dqn", "drqn", "dtqn", "adt_dqn"]

for m in models:
    plt.plot(load(m), label=m.upper())

plt.xlabel("Episode")
plt.ylabel("Return")
plt.title("MiniGrid-MemoryS9 Learning Curves")
plt.legend()
plt.grid()
plt.show()

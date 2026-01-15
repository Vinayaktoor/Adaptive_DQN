from train import run
from models.dqn import DQN
from models.drqn import DRQN
from models.dtqn import DTQN
from models.adt_dqn import ADTDQN
import torch
from utils.trainer import train_dqn, train_adt

models = {
    "DQN": DQN,
    "DRQN": DRQN,
    "DTQN": DTQN,
    "ADT-DQN": ADTDQN
}

results = {}

for name, cls in models.items():
    model = cls(obs_dim=4, action_dim=2)
    target = cls(obs_dim=4, action_dim=2)
    if name == "ADT-DQN":
        rewards, depths = run(model, target, train_adt)
        results[name] = (sum(rewards)/len(rewards), sum(depths)/len(depths))
    else:
        rewards, _ = run(model, target, train_dqn)
        results[name] = (sum(rewards)/len(rewards), None)

print(results)

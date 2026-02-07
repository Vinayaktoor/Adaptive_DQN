# Adaptive Depth Transformer-DQN (ADT-DQN)

This repository contains the official implementation of **Adaptive Depth Transformer-DQN (ADT-DQN)**,  
a value-based reinforcement learning algorithm that enables **state-dependent adaptive computation**
in Transformer-based Q-networks.

ADT-DQN dynamically adjusts the depth of Transformer computation on a per-state basis using
Q-value uncertainty, achieving comparable performance to fixed-depth Transformer-DQN
while significantly reducing average inference cost.

📄 **Paper:** Adaptive Depth Transformer-DQN: Dynamic Computation for Value-Based Reinforcement Learning  
🔗 **TechrXiv:**: doi.org/10.36227/techrxiv.176948800.00433159/v1 
💻 **Code:** This repository

---

## 🚀 Key Idea

Standard Transformer-based Q-networks apply a **fixed number of layers** for all states, leading to
unnecessary computation for simple decisions and insufficient flexibility for ambiguous states.

**ADT-DQN introduces:**
- Intermediate Q-value heads at each Transformer layer
- An uncertainty-aware halting mechanism
- Adaptive early exits while preserving Bellman consistency

This enables **dynamic computation depth** without sacrificing performance.

---

## 🧠 Method Overview

At each decision step:
1. The agent processes a fixed-length history using a Transformer encoder.
2. Each Transformer layer produces a Q-value estimate.
3. Q-value uncertainty is measured via action-wise variance.
4. Computation halts early when uncertainty falls below a threshold.
5. Training supervises all intermediate Q-heads using a shared Bellman target.

---

## 📊 Results Summary

ADT-DQN demonstrates:
- Comparable returns to fixed-depth Transformer-DQN
- Significantly reduced average computation depth
- Efficient state-dependent reasoning behavior

<p align="center">
  <img src="reward_comparison.png" width="45%" />
  <img src="adaptive_depth.png" width="45%" />
</p>

---

## 🧪 Implemented Baselines

The repository includes implementations of:
- **DQN** — Feedforward Deep Q-Network
- **DRQN** — Recurrent Q-Network
- **DTQN** — Fixed-depth Transformer Q-Network
- **ADT-DQN** — Adaptive Depth Transformer-DQN (proposed)

---

## 🗂 Repository Structure

Adaptive_DQN/

├── models/

│ ├── dqn.py
│ ├── drqn.py
│ ├── dtqn.py
│ └── adt_dqn.py
│
├── utils/

│ ├── replay_buffer.py
│ ├── trainer.py
│ └── metrics.py
│
├── train.py
├── compare.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

pip install -r requirements.txt
To train ADT-DQN on CartPole:
python train.py
To compare all models (DQN, DRQN, DTQN, ADT-DQN):
python compare.py

🔬 Reproducibility Notes

Fixed-length sequence padding is used for replay buffer stability

Target networks are updated periodically

Adaptive depth is disabled during early warm-up episodes

All experiments use identical training budgets and hyperparameters
📌 Citation

If you use this code or build upon this work, please cite:
@article{toor2026adtdqn,
  title   = {Adaptive Depth Transformer-DQN: Dynamic Computation for Value-Based Reinforcement Learning},
  author  = {Toor, Vinayak},
  journal = {arXiv preprint},
  year    = {2026}
}

🙏 Acknowledgements

This work builds upon prior research in:

Deep Q-Networks (Mnih et al.)

Recurrent and Transformer-based Q-learning

Adaptive computation mechanisms such as ACT and PonderNet
📬 Contact

For questions, feedback, or collaboration inquiries:

Vinayak Toor
📧 vinayak.toor@email.com

🔗 https://github.com/Vinayaktoor

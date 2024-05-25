from datetime import datetime
from pathlib import Path

from ml_logger.ml_logger import now
from scipy.fft import fft, fftshift

import torch
from tqdm import trange, tqdm

from rand_mdp import ToyMDP
import matplotlib.pyplot as plt
import numpy as np

from cmx import doc

with doc:
    def vi_by_horizon(rewards, dyn_mats, gamma=0.9, H=10):
        q_values = np.zeros(dyn_mats.shape[:2])

        for step in trange(H, desc="value iteration"):
            q_max = q_values.max(axis=0)
            q_values = rewards + gamma * (dyn_mats @ q_max)

        return q_values

doc @ """
Collect the spectrum from a collection of MDPs.
"""

if __name__ == '__main__':
    with doc:
        seed = 5
        torch.manual_seed(seed)
        mdp = ToyMDP(seed=seed, k=4, use_example=True)

        states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=200)
        q_1, q_2 = vi_by_horizon(rewards, dyn_mats, gamma=0.9, H=200)

        plt.figure(figsize=(9, 4))
        plt.subplot(121)
        plt.title('Dynamics')
        plt.plot(mdp.kinks, mdp.nodes[0], color="#5a5a6e", linestyle="--", linewidth=3, label="Action I")
        plt.plot(mdp.kinks, mdp.nodes[1], color="red", alpha=0.8, linewidth=3, label="Action II")
        # plt.legend(loc=(0.44, 0.05), handlelength=1)
        plt.legend(loc=(0.65, 0.7), handlelength=1)
        plt.ylim(-0.1, 1.4)

        plt.subplot(122)
        plt.title('Value')
        plt.plot(states, q_1, linewidth=1.5, label="Action I", color="#5a5a6e")
        plt.plot(states, q_2, linewidth=1.5, label="Action II", color='red')
        # plt.plot(states, q_1, linewidth=2, label="Action I")
        # plt.plot(states, q_2, linewidth=1.3, label="Action II")
        plt.ylim(0, 12)
        plt.legend(loc=(0.65, 0.7), handlelength=1)
        plt.tight_layout()
        doc.savefig(f'{Path(__file__).stem}/toy_mdp_{seed}.png?ts={now("%f")}')
        # plt.title("")  # remove title in PDF for use in LaTeX
        plt.savefig(f'{Path(__file__).stem}/toy_mdp_{seed}.pdf')

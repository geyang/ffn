from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from cmx import doc
from scipy.fft import fft, fftshift
from tqdm import trange, tqdm

from rand_mdp import ToyMDP

with doc:
    def vi_by_horizon(rewards, dyn_mats, gamma=0.9, H=10):
        q_values = np.zeros(dyn_mats.shape[:2])

        for step in range(H):
            q_max = q_values.max(axis=0)
            q_values = rewards + gamma * (dyn_mats @ q_max)

        return q_values

doc @ """
Collect the spectrum from a collection of MDPs.
"""

colors = ['black', '#23aaff', '#ff7575', '#66c56c', '#f4b247']

if __name__ == '__main__':
    plt.figure(figsize=(8, 4))
    plt.title('Spectrum vs Unroll Horizon')
    plt.ylim(-100, 0)

    with doc:
        for H, alpha in tqdm(zip([4, 3, 2, 1], [0.3, 0.5, 0.8, 1]), leave=False):
            centered_spectrum = 0
            for seed in trange(100):
                torch.manual_seed(seed)
                mdp = ToyMDP(seed=seed, k=10)

                states, rewards, dyn_mats = mdp.get_discrete_mdp(num_states=200)
                q_1, q_2 = vi_by_horizon(rewards, dyn_mats, gamma=1, H=H)

                spectrum = fft(q_1 / (q_1.mean() + 1e-12) - 1)
                centered_spectrum += fftshift(spectrum).__abs__()

            a = centered_spectrum / centered_spectrum.max()
            plt.plot(np.linspace(-np.pi, np.pi, 200), 10 * np.log(a), label=f"H={H}", color='#23aaff',
                     alpha=alpha, linewidth=3)

    plt.legend(frameon=False, loc=(1.05, 0.3))
    plt.tight_layout()
    plt.ylabel('dB')
    doc.savefig(f'{Path(__file__).stem}/toy_mdp_spectrum.png')
    plt.title("")  # remove title in PDF for use in LaTeX
    plt.savefig(f'{Path(__file__).stem}/toy_mdp_spectrum.pdf')
    doc.flush()

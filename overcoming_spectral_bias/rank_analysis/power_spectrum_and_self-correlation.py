import os
from pathlib import Path

import torch
import numpy as np
from cmx import doc


@torch.no_grad()
def spectral_entropy(power_spectrum):
    """
    The input needs to be the power spectrum as opposed to the spectrum.
    """
    spectral_norm = np.sum(power_spectrum)
    p = power_spectrum / spectral_norm
    H = np.sum(- np.log(p) * p)
    spectral_H = np.exp(H)
    return spectral_H


@torch.no_grad()
def self_correlation(waveform):
    l = len(waveform)
    return np.stack([np.mean(waveform[:i] + waveform[i:]) for i in range(l - 1)])


# if __name__ == '__main__':
#     print(np.log(3))
#     print(spectral_entropy([1, 1, 1]))
#     print(np.log(6))
#     print(spectral_entropy([1, 1, 1, 1, 1, 1]))
#     print(np.log(9))
#     print(spectral_entropy([1, 1, 1, 1, 1, 1, 1, 1, 1]))
#     exit()


# @torch.no_grad()
# def effective_rank(f, xs, ):
#     feats = f(xs)
#     feat_matrix = feats @ feats.T
#     sgv = torch.linalg.svdvals(feat_matrix)
#     ps = sgv / sgv.sum()
#     return torch.exp(- torch.sum(ps * torch.log(ps)))

if __name__ == '__main__':
    from ml_logger import logger
    import matplotlib.pyplot as plt

    logger.configure(root=os.getcwd())
    with doc @ "Inspect $\gamma$":
        for entry in logger.load_pkl('data/spectrum_gamma.pkl'):
            key, = entry.keys()
            spectrum, = entry.values()
            rank = spectral_entropy(spectrum ** 2)
            doc.print(key, rank)

    with doc @ "Now inspect the Horizon":
        for entry in logger.load_pkl('data/spectrum_H.pkl'):
            key, = entry.keys()
            spectrum, = entry.values()
            rank = spectral_entropy(spectrum ** 2)
            doc.print(key, rank)

    exit()
    Hs = [4, 3, 2, 1]
    H_ranks = [4.792956693917493,
               4.196312377576643,
               3.6720854877558344,
               1.054024808601584]

    plt.figure(figsize=(4, 3))
    plt.plot(Hs[::-1], H_ranks[::-1], 'o-')
    plt.title("Spectral Entropy vs Horizon")
    plt.xlabel('H')
    plt.ylabel('Entropy')
    plt.ylim(0.5, 7)
    doc.savefig(f"{Path(__file__).stem}/rank_H.png")
    plt.savefig(f"{Path(__file__).stem}/rank_H.pdf")
    plt.close()

    gammas = [0.99, 0.9, 0.6, 0.1]
    gamma_ranks = [5.164980044490921, 5.044699162041709, 4.6090311418051995, 2.3659132117522814]
    plt.figure(figsize=(4, 3))
    plt.title("Spectral Entropy vs Gamma")
    plt.plot(gamma_ranks[::-1], 'o-')
    plt.xticks(ticks=range(4), labels=gammas[::-1])
    plt.ylabel('Entropy')
    plt.xlabel('$\gamma$')
    plt.ylim(0.5, 7)
    doc.savefig(f"{Path(__file__).stem}/rank_gamma.png")
    plt.savefig(f"{Path(__file__).stem}/rank_gamma.pdf")

    doc.flush()

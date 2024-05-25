from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cmx import doc
from ml_logger import logger
from tqdm import tqdm, trange

from simple_ntk.models import MLP

if __name__ == '__main__':
    doc @ """
    # How does the row sum scale with Gaussian?
    """
    with doc:
        np.random.seed(100)
        torch.random.manual_seed(100)

        dims = np.arange(1, 101)
        samples = 100

    doc @ """
    The scale of the matrix
    $$
    \\vert W \\vert
    $$
    determines the width of the NTK. We can compute the bandwidth
    parameters needed to compensate for the growth in dimension via
    the following:
    """
    with doc:
        def get_row_scale(pdf, n_dim, samples):
            return pdf(n_dim, [samples, n_dim]).abs().mean(dim=0).sum().item()


        siren_pdf = lambda dim, *_: torch.normal(0, 1 / np.sqrt(dim), *_)
        our_pdf = lambda dim, *_: torch.normal(0, 1 / dim, *_)

        for d in dims:
            s = get_row_scale(siren_pdf, d, samples)
            o = get_row_scale(our_pdf, d, samples)
            logger.store_metrics(
                siren_B_abs=s,
                our_B_abs=o,
                siren_scaling=1 / s,
                our_scaling=1 / o
            )

    r = doc.table().figure_row()
    plt.figure(figsize=(5, 4))
    plt.plot(dims, logger.summary_cache['siren_B_abs'],
             color="gray", linewidth=2.0, label="SIREN $b/\sqrt{n}$")
    plt.plot(dims, logger.summary_cache['our_B_abs'],
             color="#23aaff", linewidth=2.0, label="Ours $b/n$")
    plt.yscale('log')
    plt.ylim(0.3, 50)
    plt.ylabel('$\\vert B\\vert$')
    plt.xlabel('Dimension $n$')
    plt.legend()
    plt.title('Kernel $\\vert B \\vert$ Scaling')
    r.savefig(f"{Path(__file__).stem}/row_weight_vs_dim.png")
    plt.savefig(f"{Path(__file__).stem}/row_weight_vs_dim.pdf")
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(dims, logger.summary_cache['our_scaling'],
             color="#23aaff", linewidth=2.0, label="Ours $b/n$")
    plt.plot(dims, logger.summary_cache['siren_scaling'],
             color="gray", linewidth=2.0, label="SIREN $b/\sqrt{n}$")
    plt.yscale('log')
    plt.ylim(4e-2, 3)
    plt.ylabel('Parameter $b$')
    plt.xlabel('Dimension $n$')
    plt.legend(loc='lower center', ncol=2)
    plt.title('Parameter $b$ Scaling')
    r.savefig(f"{Path(__file__).stem}/parameter_scaling.png")
    plt.savefig(f"{Path(__file__).stem}/parameter_scaling.pdf")

    doc.flush()

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from cmx import doc
from tqdm import tqdm, trange

from simple_ntk import get_ntk
from simple_ntk.models import RFF, MLP

if __name__ == "__main__":
    doc @ """
    # Reproducing NTK kernel of the Fourier Features
    
    This notebook uses the `simple_ntk` module available in this directory
    to reproduce Fig 2. from the paper
    
    [^fourier]: 
    
    """
    with doc:
        xs = np.linspace(-0.5, 0.5, 32)

        for p in tqdm([0, 0.5, 1, 1.5, 2, float("inf")], desc="kernels"):
            # average from five networks
            ntk_kernel = 0
            for i in trange(10, desc="averaging networks", leave=False):
                net = nn.Sequential(
                    RFF(1, 32, p),
                    MLP(32, 1024, 4, 1),
                )
                ntk_kernel += 0.1 * get_ntk(net, xs)

            plt.figure("cross section")
            plt.plot(ntk_kernel[16], label=f"p={p}")

            plt.figure("spectrum")
            fft = np.fft.fft(ntk_kernel[16])
            plt.plot(np.fft.fftshift(fft).__abs__(), label=f"p={p}")

    plt.figure("kernel")
    with doc:
        plt.imshow(ntk_kernel, cmap="inferno")

    r = doc.table().figure_row()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk.png", zoom="50%", title="Kernel", fig=plt.figure("kernel"))

    plt.figure("cross section")
    plt.legend(loc=(1, -0.05), frameon=False)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_cross.png", title="Cross Section", zoom="50%")

    plt.figure("spectrum")
    plt.yscale("log")
    plt.legend(loc=(1, -0.05), frameon=False)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_spectrum.png", title="Spectrum", zoom="50%")

    doc.flush()

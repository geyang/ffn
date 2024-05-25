from pathlib import Path
from time import sleep

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
    # Band Tunnability with Random Fourier Features
    
    An important feature is the ability to fine-tune the frequency band to match that of the target function.
    """
    with doc:
        np.random.seed(100)
        torch.random.manual_seed(100)


        class RFF(nn.Linear):
            def __init__(self, in_features, mapping_size, band_limit: int):
                super().__init__(in_features, mapping_size)
                nn.init.uniform_(self.weight, - band_limit / np.sqrt(in_features), band_limit / np.sqrt(in_features))
                nn.init.uniform_(self.bias, -1, 1)
                self.requires_grad_(False)

            def forward(self, x):
                z = super().forward(x)
                return torch.sin(np.pi * z)

    with doc:
        def get_ntk(net, xs):
            grad = []
            out = net(torch.FloatTensor(xs)[:, None])
            for o in tqdm(out, desc="NTK", leave=False):
                net.zero_grad()
                o.backward(retain_graph=True)
                grad_vec = torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None]).numpy()
                grad.append(grad_vec / np.linalg.norm(grad_vec))
                net.zero_grad()

            grad = np.stack(grad)
            gram_matrix = grad @ grad.T
            return gram_matrix

    # with doc:
    #     band_limits = [1, 3, 5, 10]
    #
    #     logger.remove('data/cross_section.pkl')
    #     xs = np.linspace(-0.5, 0.5, 32)
    #     for band_limit in tqdm(band_limits, desc="kernels"):
    #         ntk_kernel = 0  # average from ten networks
    #         for i in trange(20, desc="averaging networks", leave=False):
    #             net = nn.Sequential(RFF(1, 32, band_limit), MLP(32, 1024, 4, 1))
    #             ntk_kernel += 0.05 * get_ntk(net, xs)
    #         logger.save_pkl(ntk_kernel[16], 'data/cross_section.pkl', append=True)

    sleep(1.0)
    band_limits = [1, 3, 5, 10]
    cross_section = logger.load_pkl('data/cross_section.pkl')

    for b_highlight in band_limits:

        # colors = ['#23aaff', '#9799f9', '#d784db', '#f975ab', '#ff7777'][::-1]
        colors = ['red'] * 4

        for row, band_limit, c in zip(cross_section, band_limits, colors):
            plt.figure('cross section')
            c = c if b_highlight == band_limit else '#efefef'
            plt.plot(row, label=f'b={band_limit}', linewidth=3, alpha=0.8, color=c)

        for row, band_limit, c in list(zip(cross_section, band_limits, colors))[::-1]:
            plt.figure('spectrum')
            fft = np.fft.fft(row)
            fft_mag = np.fft.fftshift(fft).__abs__()
            c = c if b_highlight == band_limit else '#efefef'
            plt.plot(10 * np.log10(fft_mag), label=f'b={band_limit}', linewidth=3, alpha=0.8, color=c)

        r = doc.table().figure_row()

        plt.figure('cross section')
        plt.title('Cross Section')
        plt.legend(loc=(1, 0.25))
        plt.tight_layout()
        r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_cross_b_{b_highlight}.png", title="Cross Section", zoom="50%")
        plt.savefig(f"{Path(__file__).stem}/fourier_features_ntk_cross_{b_highlight}.pdf")

        plt.figure('spectrum')
        plt.title('Spectrum')
        plt.legend(loc=(1, 0.25))
        plt.ylabel('dB')
        plt.tight_layout()
        r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_spectrum_b_{b_highlight}.png", title="Spectrum", zoom="50%")
        plt.savefig(f"{Path(__file__).stem}/fourier_features_ntk_spectrum_b_{b_highlight}.pdf")

        plt.close('all')

    doc.flush()

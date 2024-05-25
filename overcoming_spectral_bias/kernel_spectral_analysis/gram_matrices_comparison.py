from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cmx import doc
from tqdm import tqdm, trange

from simple_ntk.models import MLP, D2RLNet

if __name__ == '__main__':
    doc @ """
    # Gram Matricies of Different Network Architectures
    
    We visualize the gram matrix of the following architectures
    1. A piece-wise linear network: MLP with ReLU Activation
    2. An MLP with tahn activation
    3. A dense Network~\citep{}
    4. A Fourier Feature Network
    """
    with doc:
        np.random.seed(100)
        torch.random.manual_seed(100)

    with doc:
        class FF(nn.Module):
            def __init__(self, band_limit: int, p: float):
                self.b_s = torch.arange(1, band_limit + 1)
                self.a_s = 1 / torch.pow(self.b_s, p)
                super().__init__()

            def forward(self, x):
                return torch.cat([
                    self.a_s * torch.sin(2. * np.pi * x * self.b_s),
                    self.a_s * torch.cos(2. * np.pi * x * self.b_s)
                ], dim=-1) / torch.norm(self.a_s)

    doc @ """
    Now there are a few crucial points. First of all, there are
    these two parameters `a` and `b`. In fourier features, the
    parameter `b` corresponds to the frequencies. Fourier features
    use equally spaced octaves.
    """
    with doc:
        def get_ntk(net, xs):
            grad = []
            out = net(torch.FloatTensor(xs)[:, None])
            for o in tqdm(out, desc="NTK", leave=False):
                net.zero_grad()
                o.backward(retain_graph=True)
                grad_vec = torch.cat([p.grad.view(-1) for p in net.parameters() if p.grad is not None]).numpy()
                # grad.append(grad_vec / np.linalg.norm(grad_vec))
                grad.append(grad_vec)
                net.zero_grad()

            grad = np.stack(grad)
            gram_matrix = grad @ grad.T
            return gram_matrix

    xs = np.linspace(-0.5, 0.5, 32)
    box = [-0.5, 0.5, -0.5, 0.5]

    r = doc.table().figure_row()

    # Kernel of an MLP
    plt.figure()
    ntk_kernel = 0  # average from ten networks
    for i in trange(10, desc="averaging networks", leave=False):
        net = MLP(1, 1024, 4, 1)
        ntk_kernel += 0.1 * get_ntk(net, xs)

    plt.imshow(ntk_kernel, cmap='inferno', extent=box)

    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/mlp_ntk_relu.png", zoom="80%", title="MLP + ReLU")
    plt.savefig(f"{Path(__file__).stem}/mlp_ntk_relu.pdf")

    # Kernel of an MLP with Tanh
    plt.figure()
    ntk_kernel = 0  # average from ten networks
    for i in trange(10, desc="averaging networks", leave=False):
        net = MLP(1, 1024, 4, 1, nn.Tanh)
        ntk_kernel += 0.1 * get_ntk(net, xs)

    plt.imshow(ntk_kernel, cmap='inferno', extent=box)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/mlp_ntk_tahn.png", zoom="80%", title="MLP + Tahn")
    plt.savefig(f"{Path(__file__).stem}/mlp_ntk_tahn.pdf")

    # Kernel of a dense network
    plt.figure()
    ntk_kernel = 0  # average from ten networks
    for i in trange(10, desc="averaging networks", leave=False):
        net = D2RLNet(1, 1024, 4, 1)
        ntk_kernel += 0.1 * get_ntk(net, xs)

    plt.imshow(ntk_kernel, cmap='inferno', extent=box)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/d2rl_ntk.png", zoom="80%", title="DenseNet (D2RL)")
    plt.savefig(f"{Path(__file__).stem}/d2rl_ntk.pdf")


    # Kernel of an RFF-MLP
    plt.figure()
    ntk_kernel = 0  # average from ten networks
    for i in trange(10, desc="averaging networks", leave=False):
        net = nn.Sequential(FF(16, p=1), MLP(32, 1024, 4, 1))
        ntk_kernel += 0.1 * get_ntk(net, xs)

    plt.imshow(ntk_kernel, cmap='inferno', extent=box)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/ff_mlp_ntk.png", zoom="80%", title="FF + MLP")
    plt.savefig(f"{Path(__file__).stem}/ff_mlp_ntk.pdf")

    doc.flush()

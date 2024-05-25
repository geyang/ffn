from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from cmx import doc
from tqdm import tqdm, trange

from simple_ntk.models import MLP

if __name__ == '__main__':
    doc @ """
    # A Quick Introduction to Fourier Kernels
    
    First of all, let's take a look at the implementation:
    """
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
        net = FF(8, 0)
        doc.print("frequencies (in 2π):", net.b_s)

    doc @ """
    An important property in the examples used in [^tanick] is that
    if your input is between $[0, 1)$, you need to make sure that
    the frequencies $\mathbb b$ captures the lowest mode (the first
    octave) of your entire range. Otherwise you will get aliasing.
    
    ### Important Considerations

    So for the range $[0, 1)$, this means the lowest mode spans the 
    full $2 \\pi$ over that range. 

    Now suppose we only take 1 fourier component (both $\sin$ and
    $\cos$), this range $[0, 1)$ is mapped to a circle in the
    phase space. If the a data point falls **outside** of this range,
    it would circle back in the circle and become aliased. Therefore
    we need to make sure the lowest frequency components has longer
    wavelength than the range.
    
    ## Controlling the Spectral Bias
    
    Now a second parameter is $a$. This is the set of weights
    for each of the fourier components in 
    $$
       \\text{out} = \sum {a_i \sin(2\\pi b_i x) + a_i \cos(2\\pi b_i x)}
    $$
    We can use 
    $$
       a_i = 1 / b_i^p
    $$
    to specify these weights. The weight decays for different $p$ as
    below:
    """
    plt.figure(figsize=(5, 3))
    with doc:
        plt.title("Spectral Weights")

        for p in [0, 0.5, 1, 1.5, 2, float('inf')]:
            a_s = 1 / np.arange(1, 8).__pow__(p)
            plt.plot(a_s, label=f"p={p}")

    plt.legend(loc=(1, -0.05))
    doc.savefig(f"{Path(__file__).stem}/spectral_weights.png", zoom="20%")

    doc @ """
    ## The Spectral Bias of Fourier Features
    
    We can directly visualize the spectrum of this kernel. Now before we start
    let's first remind ourselves of the basic concepts. 
    
    **definition 1:** a similarity function $K: X \mapsto \mathbb R$ that is 
    *continuous* and *symmetric*, is called a *kernel*.
    
    **Mercer's Theorem** $K$ is a *kernel* **iff.** for *all* data 
    $x\\in X$, the Gramian matrix $M$ given by 
    $$
    G_{[i, j]} = K\\langle x^i, x^j\\rangle
    $$
    is positive semi-definite (PSD). This is called the *kernel
    trick*. For more precise formalism, refer to wikipedia.
    
    
    The neural tangent kernel (NTK) is similar to the *conjugate kernel* (CK). The congjugate kernel looks at the neural network at initialization, whereas the neural tanget kernel looks at the *gradient* of a network initialized around *zero*. This is related to the first order Taylor expansion of the neural network function, and can be done by computing the Gramian matrix of the gradient vector around the intialization point $\\theta_0$ of the network $f_{\\theta_0}(x)$:
    $$
    NTK\langle x^i, x^j \\rangle = \\nabla_\\theta f(x^i) \\nabla^T_\\theta f(x^j)
    $$
    There are two important details:

    1. The gradient vector $\\nabla_\\theta f$ needs to be normalized, so that for the same datapoint $x_i = x_j$, the NTK kernel produces the identity.

       This is the reason behind the line below

       ```python
       grad.append(grad_vec / np.linalg.norm(grad_vec))
       ```

    2. The Gram matrix for one network instantiation can be quite noisy. We need to average over multiple instantiations.
    """
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

    with doc:
        xs = np.linspace(-0.5, 0.5, 32)

        for p in tqdm([0, 0.5, 1, 1.5, 2, float('inf')], desc="kernels"):
            ntk_kernel = 0  # average from ten networks
            for i in trange(10, desc="averaging networks", leave=False):
                net = nn.Sequential(FF(16, p), MLP(32, 1024, 4, 1))
                ntk_kernel += 0.1 * get_ntk(net, xs)

            plt.figure('cross section')
            plt.plot(ntk_kernel[16], label=f'p={p}')

            plt.figure('spectrum')
            fft = np.fft.fft(ntk_kernel[16])
            plt.plot(np.fft.fftshift(fft).__abs__(), label=f'p={p}')

    plt.figure('kernel')
    with doc:
        plt.imshow(ntk_kernel, cmap='inferno')

    r = doc.table().figure_row()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk.png", zoom="50%",
              title="Kernel", fig=plt.figure('kernel'))

    plt.figure('cross section')
    plt.legend(loc=(1, -0.05), frameon=False)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_cross.png",
              title="Cross Section", zoom="50%")

    plt.figure('spectrum')
    plt.yscale('log')
    plt.legend(loc=(1, -0.05), frameon=False)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_spectrum.png",
              title="Spectrum", zoom="50%")

    doc @ """
    ## The Band-limited Fourier Features
    
    An important feature of our implementation above, is that the
    fourier features are band-limited on the top. For `band_limit=16`,
    if we visualize *beyond* the 8th octave, we would see the spectrum
    encounter a sharp drop. 
    
    This happens because we are effectively clipping the higher order
    components
    """

    with doc:
        plt.figure("Full A's")
        for p in [0, 0.5, 1, 1.5, 2, float('inf')]:
            ff = FF(16, p)
            full_a = np.zeros(32)
            full_a[:16] = ff.a_s
            plt.plot(full_a, label=f"p={p}")

    with doc:
        xs = np.linspace(-0.5, 0.5, 64)

        for p in tqdm([0, 0.5, 1, 1.5, 2, float('inf')], desc="kernels"):
            ntk_kernel = 0
            for i in trange(10, desc="averaging networks", leave=False):
                net = nn.Sequential(FF(16, p), MLP(32, 1024, 4, 1))
                ntk_kernel += 0.1 * get_ntk(net, xs)

            plt.figure('Full Spectrum')
            fft = np.fft.fft(ntk_kernel[32])
            plt.plot(np.fft.fftshift(fft).__abs__(), label=f'p={p}')

    r = doc.table().figure_row()
    plt.figure("Full A's")
    plt.ylim(-0.1, 1.2)
    r.savefig(f"{Path(__file__).stem}/full_a_s.png",
              title="Full A's", zoom="50%")

    plt.figure('Full Spectrum')
    plt.yscale('log')
    plt.legend(loc=(1, -0.05), frameon=False)
    plt.tight_layout()
    r.savefig(f"{Path(__file__).stem}/fourier_features_ntk_full_spectrum.png",
              title="Spectrum", zoom="50%")

    doc.flush()

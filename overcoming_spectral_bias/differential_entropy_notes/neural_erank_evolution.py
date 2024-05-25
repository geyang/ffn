from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from cmx import doc

if __name__ == '__main__':
    doc @ """
    # S-Rank of A Neural Network During Evolution
    
    """
    with doc:
        import torch
        from torch import nn
        from torch import optim
        from tqdm import trange
        from ml_logger import logger
        from simple_ntk.models import RFF

    with doc, doc.table().figure_row() as r:
        N = 1001
        b = 20
        xs = np.linspace(0, 5, N)
        bs = np.linspace(-1, 1, N)
        spectrum = 2 * np.pi * np.arange(b)
        target_fn = np.sum([np.sin(k * xs + np.pi * bs) for k in spectrum], axis=0)
        plt.plot(xs, target_fn)
        r.savefig(f"{Path(__file__).stem}/target_fn.png", zoom="50%")

    plt.close()

    with doc:
        def feat(network, xs):
            *layers, last = network
            for l in layers:
                xs = l(xs)
            return xs


        def H_d(ps):
            ps_norm = ps / ps.sum()
            return - np.sum(np.log(ps) * ps_norm)


        @torch.no_grad()
        def H_srank(xs, feat_fn, net):
            zs = feat_fn(net, xs)
            gram_matrix = zs @ zs.T
            sgv = torch.linalg.svdvals(gram_matrix)
            sgv /= sgv.sum()
            return H_d(sgv.cpu().numpy())


    def train(net, prefix):
        xs_t = torch.Tensor(xs)[..., None]
        target_t = torch.Tensor(target_fn)[..., None]
        adam = optim.Adam(net.parameters(), lr=1e-3)
        for epoch in trange(200):
            output = net(xs_t)
            loss = nn.functional.smooth_l1_loss(output, target_t)

            adam.zero_grad()
            loss.backward()
            adam.step()

            if epoch % 20 == 0:
                print(f"\repoch{epoch} loss:", loss.detach().cpu().item())
                srank = H_srank(xs_t, feat, net)

                # evaluation logic here
                with logger.Prefix(metrics=prefix):
                    logger.store_metrics(epoch=epoch, srank=srank)

        return output.detach().cpu().numpy()


    def visualize(fitted, prefix):
        color = colors.pop(0)

        plt.figure('fit')
        plt.plot(xs, fitted, color=color, label=prefix)

        plt.figure('effective rank')
        plt.plot(logger.summary_cache[f'{prefix}/epoch'], logger.summary_cache[f'{prefix}/srank'],
                 color=color, label=prefix)


    doc @ """
    ## How does Architecture Affect The Rank?
    
    """
    with doc:
        network = nn.Sequential(
            RFF(1, 40, scale=10),
            nn.Linear(40, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 1),
        )
        fitted = train(network, prefix="RFF")
        colors = ['#23aaff', '#ff7777']
        visualize(fitted, prefix="RFF")

        network = nn.Sequential(
            nn.Linear(1, 40), nn.ReLU(),
            nn.Linear(40, 20), nn.ReLU(),
            nn.Linear(20, 20), nn.ReLU(),
            nn.Linear(20, 1),
        )
        fitted = train(network, prefix="MLP")
        visualize(fitted, prefix="MLP")

    r = doc.table().figure_row()
    plt.figure('fit')
    plt.plot(xs, target_fn, color='gray', label="target")
    plt.legend(frameon=False)
    r.savefig(f"{Path(__file__).stem}/fit.png", zoom="50%", caption="Fit Result")
    plt.close()

    plt.figure('effective rank')
    plt.legend(frameon=False)
    r.savefig(f"{Path(__file__).stem}/s_rank.png", zoom="50%", caption="S-Rank")
    plt.close()

    doc @ """
    ## How does B_scale Affect the Rank?
    """
    with doc:
        colors = ['#23aaff', '#9799f9', '#d784db', '#ff7777', "orange"]
        for scale in [2, 5, 10, 20, 30]:
            network = nn.Sequential(
                RFF(1, 40, scale=10),
                nn.Linear(40, 20), nn.ReLU(),
                nn.Linear(20, 20), nn.ReLU(),
                nn.Linear(20, 1),
            )
            fitted = train(network, prefix=f"RFF {scale}")
            visualize(fitted, prefix=f"RFF {scale}")

    r = doc.table().figure_row()
    plt.figure('fit')
    plt.plot(xs, target_fn, color='gray', label="target")
    plt.legend(frameon=False)
    r.savefig(f"{Path(__file__).stem}/fit_scale.png", zoom="50%", caption="Fit Result")

    plt.figure('effective rank')
    plt.legend(frameon=False)
    r.savefig(f"{Path(__file__).stem}/s_rank_scale.png", zoom="50%", caption="S-Rank")

    doc.flush()

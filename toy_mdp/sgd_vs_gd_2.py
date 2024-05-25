import os

import numpy as np
import torch.nn as nn
import torch.optim
from cmx import doc
from tqdm import trange


def plot_value(xs, ys, ground_truth, losses, fig_prefix, title=None, doc=doc):
    import matplotlib.pyplot as plt

    plt.plot(xs, ys, color="#23aaff", label="fit")
    plt.plot(xs, ground_truth, color="gray", label="Ground Truth")
    if title:
        plt.title(title)
    plt.legend()
    plt.xlabel('State [0, 1)')
    plt.ylabel('Value')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}.png?ts={doc.now("%f")}', dpi=300, zoom=0.3)
    plt.close()

    plt.plot(losses)
    plt.hlines(0, 0, len(losses), linestyle='--', color='gray')
    plt.title("Loss")
    plt.xlabel('Optimization Steps')
    doc.savefig(f'{os.path.basename(__file__)[:-3]}/{fig_prefix}_loss.png?ts={doc.now("%f")}', dpi=300,
                zoom=0.3)
    plt.close()


class Buffer:
    def __init__(self, *data, batch_size=None):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data[0])

    def sample(self, batch_size):
        inds = torch.rand(size=(self.__len__(),)).argsort()
        from more_itertools import chunked
        for batch_inds in chunked(inds, n=batch_size):
            yield [torch.Tensor(d[batch_inds]) for d in self.data]


def supervised(xs, ys, lr=4e-4, n_epochs=100, batch_size=None):
    # Ge: need to initialize the Q function at zero
    f = nn.Sequential(
        nn.Linear(1, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

    optim = torch.optim.RMSprop(f.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    dataset = Buffer(xs[..., None], ys)

    losses = []
    for epoch in trange(n_epochs + 1, desc=f"batch_size={batch_size or 'all'}"):
        for x_batch, y_batch in dataset.sample(batch_size=batch_size or len(dataset)):
            y_bar = f(x_batch)
            loss = l1(y_bar, y_batch)
            losses.append(loss.detach().numpy())

            optim.zero_grad()
            loss.backward()
            optim.step()

    with torch.no_grad():
        all_xs = torch.Tensor(xs[..., None])
        return f(all_xs).squeeze().numpy().T, losses


class RFF(nn.Module):
    """
    get self.B.std_mean()
    """

    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = mapping_size * 2
        self.B = torch.normal(0, scale, size=(mapping_size, self.input_dim))
        self.B.requires_grad = False

    def forward(self, x):
        x = 2 * np.pi * x @ self.B.T
        return torch.cat([torch.cos(x), torch.sin(x)], dim=1)


def supervised_rff(states, values, dyn_mats, lr=1e-4, gamma=0.9, n_epochs=100, B_scale=1):
    # Ge: need to initialize the Q function at zero
    Q = nn.Sequential(
        RFF(1, 200, scale=B_scale),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 400),
        nn.ReLU(),
        nn.Linear(400, 2),
    )

    optim = torch.optim.RMSprop(Q.parameters(), lr=lr)
    l1 = nn.functional.smooth_l1_loss

    states = torch.FloatTensor(states).unsqueeze(-1)
    values = torch.FloatTensor(values)

    losses = []

    for epoch in trange(n_epochs + 1):
        values_bar = Q(states)
        loss = l1(values_bar, values.T)
        losses.append(loss.detach().numpy())

        optim.zero_grad()
        loss.backward()
        optim.step()

    q_values = values_bar.T.detach().numpy()
    return q_values, losses


if __name__ == "__main__":
    doc @ """
    # Spectral Bias
    
    here is a random superposition of 10 fourier components (random phase).
    """

    with doc:
        xs = np.linspace(0, 1, 401)
        ys = np.stack([np.sin(np.random.random() + 2 * np.pi * k * xs) for k in range(5, 55, 5)]).sum(axis=0)

    with doc:
        ys_bar, losses = supervised(xs, ys[..., None], batch_size=32, n_epochs=10_000)

    plot_value(xs, ys_bar[0], ys, losses, fig_prefix="spectral_bias_sgd",
               title="Supervised with SGD", doc=doc.table().figure_row())

    doc @ """
    ## Supervised Learning with MLP and RFF
    
    Here is the ground truth value function generated via tabular
    value iteration. It shows even for simple dynamics, the value
    function can be exponentially more complex.
    """
    from matplotlib import pyplot as plt

    with doc:
        states = np.loadtxt("data/states.csv", delimiter=',')
        gt_q_values = np.loadtxt("data/q_values.csv", delimiter=',')

    doc @ """
    ## Supervised Learning with SGD
    
    Here we use the a batch size of 32
    """
    with doc:
        q_values, losses = supervised(states, gt_q_values.T, batch_size=32, n_epochs=10_000)

    plot_value(states, q_values[0], gt_q_values[0], losses, fig_prefix="supervised_sgd",
               title="Supervised with SGD", doc=doc.table().figure_row())

    doc @ """
    ## Supervised Learning with GD
    
    Here we use the entire dataset as a single batch (GD).
    """
    with doc:
        q_values, losses = supervised(states, gt_q_values.T, n_epochs=120_000)

    plot_value(states, q_values[0], gt_q_values[0], losses, fig_prefix="supervised_gd",
               title="Supervised with GD", doc=doc.table().figure_row())

    doc.flush()

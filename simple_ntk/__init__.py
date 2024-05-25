import numpy as np
import torch
from tqdm import tqdm


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

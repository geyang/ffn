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
    # Investigate the Weight Initialization Scheme of D2RL Network
    
    """
    with doc:
        np.random.seed(100)
        torch.random.manual_seed(100)

    with doc:
        net = D2RLNet(1, 1024, 4, 1)
        for p in net.parameters():
            doc.print(p.shape, p.max().item(), p.min().item())
    doc.flush()

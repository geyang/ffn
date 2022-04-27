import numpy as np
import torch
import torch.nn as nn

class CLFF(nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, in_channel, out_channel, scale=1.0, kernel_size=1, stride=1, init="iso", sincos=False):
        super().__init__()
        self.sincos = sincos
        self.scale = scale
        if self.sincos:
            self.cff = nn.Conv2d(in_channel, out_channel//2, kernel_size, stride=stride)
        else:
            self.cff = nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride)
        if init == "iso":
            nn.init.normal_(self.cff.weight, 0, scale / in_channel)
            nn.init.normal_(self.cff.bias, 0, 1)
        else:
            nn.init.uniform_(self.cff.weight, -scale / in_channel, scale / in_channel)
            nn.init.uniform_(self.cff.bias, -1, 1)

        if self.sincos:
            nn.init.zeros_(self.cff.bias)

    def forward(self, x, **_):
        x = np.pi * self.cff(x)
        if self.sincos:
            # Assumes x = (N, C, H, W)
            return torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        else:
            return torch.sin(x)

    def copy_weights_from(self, source):
        """Tie layers"""
        from ml_logger import logger
        self.cff.weight = source.cff.weight
        self.cff.bias = source.cff.bias

class CRFF(CLFF):
    def __init__(self, in_channel, out_channel, scale=1.0, kernel_size=1, stride=1, init="iso", sincos=False):
        super().__init__(in_channel, out_channel, scale, kernel_size, stride, init, sincos)
        self.cff.requires_grad = False

class LFF(nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, in_features, out_features, scale=1.0, init="iso", sincos=False):
        super().__init__()
        self.in_features = in_features
        self.sincos = sincos
        self.out_features = out_features
        self.scale = scale
        if self.sincos:
            self.linear = nn.Linear(in_features, self.out_features//2)
        else:
            self.linear = nn.Linear(in_features, self.out_features)
        if init == "iso":
            nn.init.normal_(self.linear.weight, 0, scale / self.in_features)
            nn.init.normal_(self.linear.bias, 0, 1)
        else:
            nn.init.uniform_(self.linear.weight, -scale / self.in_features, scale / self.in_features)
            nn.init.uniform_(self.linear.bias, -1, 1)
        if self.sincos:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x, **_):
        x = np.pi * self.linear(x)
        if self.sincos:
            return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        else:
            return torch.sin(x)

    def copy_weights_from(self, source):
        """Tie layers"""
        from ml_logger import logger
        self.linear.weight = source.linear.weight
        self.linear.bias = source.linear.bias

class RFF(LFF):
    def __init__(self, in_features, out_features, scale=1.0, **kwargs):
        super().__init__(in_features, out_features, scale=scale, **kwargs)
        self.linear.requires_grad = False


class RFF_tanick(RFF):
    """
    Original Random Fourier Features Implementation from Tanick et al.
    - Tancik, M., Srinivasan, P. P. and Mildenhall, B. (2020) ‘Fourier Features Let Networks Learn
      High Frequency Functions In Low Dimensional Domains’, arXiv preprint arXiv.
      Available at: https://arxiv.org/abs/2006.10739.
    """
    def __init__(self, in_features, out_features, scale=1.0, **kwargs):
        super().__init__(in_features, out_features, scale=scale, init="iso", sincos=True)

class SIREN(LFF):
    def __init__(self, in_features, out_features, scale=1.0, **kwargs):
        super().__init__(in_features, out_features, scale=scale, init="unif")

RFF_dict = {
    'lff': LFF,
    'rff': RFF,
    'rff_tanick': RFF_tanick,
    'siren': SIREN,
}
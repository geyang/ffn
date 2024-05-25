import numpy as np
import torch
from torch import nn
from torch.nn import functional


class FF(nn.Module):
    """
    This is the 1-D Fourier Mapping limited to 1-dimension.
    """

    def __init__(self, band_limit: int, p: float):
        self.out_features = band_limit * 2
        self.p = p
        self.b_s = torch.arange(1, band_limit + 1)
        self.a_s = 1 / torch.pow(self.b_s, p)
        super().__init__()

    def forward(self, x):
        return torch.cat([
            self.a_s * torch.sin(2. * np.pi * x * self.b_s),
            self.a_s * torch.cos(2. * np.pi * x * self.b_s)
        ], dim=-1) / torch.norm(self.a_s)


class LFF(nn.Module):
    """
    get torch.std_mean(self.B)
    """

    def __init__(self, in_features, out_features, scale=1.0, init="iso"):
        super().__init__()
        self.input_dim = in_features
        self.output_dim = out_features
        self.linear = nn.Linear(in_features, self.output_dim)

        if init == "iso":
            nn.init.normal_(self.linear.weight, 0, scale / self.input_dim)
        else:
            nn.init.uniform_(self.linear.weight, -scale / self.input_dim, scale / self.input_dim)

        nn.init.uniform_(self.linear.bias, -1, 1)

    def forward(self, x):
        x = np.pi * self.linear(x)
        return torch.sin(x)


class RFF(LFF):
    def __init__(self, input_dim, mapping_size, scale=1, **kwargs):
        super().__init__(input_dim, mapping_size, scale=scale, **kwargs)
        self.linear.requires_grad = False


class RFF_tanick(RFF):
    """
    Original Random Fourier Features Implementation from Tanick et al.

    - Tancik, M., Srinivasan, P. P. and Mildenhall, B. (2020) ‘Fourier Features Let Networks Learn
      High Frequency Functions In Low Dimensional Domains’, arXiv preprint arXiv.
      Available at: https://arxiv.org/abs/2006.10739.
    """

    def __init__(self, input_dim, mapping_size, scale=1):
        super().__init__(input_dim, mapping_size, scale=scale, init="iso")


class LFN(nn.Sequential):
    """
    Learned Fourier Networks
    """

    def __init__(self, in_features, out_features, in_scale, latent_scale,
                 latent_dims=(40, 1000, 10000)):
        layers = []
        for d0, d1 in zip(latent_dims[:-1], latent_dims[1:]):
            layers += [LFF(d0, d1, scale=latent_scale)]

        super().__init__(
            LFF(in_features, latent_dims[0], scale=in_scale),
            *layers,
            nn.Linear(latent_dims[-1], out_features),
        )


class LFF_MLP(nn.Sequential):
    def __init__(self, in_features, out_features, scale,
                 latent_dims=(400, 400, 400), activation=nn.ReLU):
        layers = []
        for d0, d1 in zip(latent_dims[:-1], latent_dims[1:]):
            layers += [nn.Linear(d0, d1), activation()]

        super().__init__(
            LFF(in_features, latent_dims[0], scale=scale),
            *layers,
            nn.Linear(latent_dims[-1], out_features),
        )


class RFN(nn.Sequential):
    """
    Random Fourier Networks
    """

    def __init__(self, in_features, out_features, scale,
                 latent_dims=(400, 400, 400), activation=nn.ReLU):
        layers = []
        for d0, d1 in zip(latent_dims[:-1], latent_dims[1:]):
            layers += [nn.Linear(d0, d1), activation()]

        super().__init__(
            RFF(in_features, latent_dims[0], scale=scale),
            *layers,
            nn.Linear(latent_dims[-1], out_features),
        )


# @torch.no_grad()
class MLP(nn.Sequential):
    def __init__(self, in_features, latent_features, latent_layers, out_features, activation=nn.ReLU):
        self.in_features = in_features
        self.latent_features = latent_features
        self.latent_layers = latent_layers
        self.out_features = out_features

        layers = [nn.Linear(in_features, latent_features), activation()]
        for _ in range(latent_layers - 1):
            layers += [nn.Linear(latent_features, latent_features), activation()]
        layers += [nn.Linear(latent_features, out_features)]

        super().__init__(*layers)


class D2RLNet(nn.Sequential):
    def __init__(self, in_features, latent_features, latent_layers, out_features):
        layers = [nn.Linear(in_features, latent_features)]
        for _ in range(latent_layers - 1):
            layers += [nn.Linear(latent_features + in_features, latent_features)]
        layers += [nn.Linear(latent_features + in_features, out_features)]
        super().__init__(*layers)
        # Override the default, which is 1/sqrt(in_dim)
        for p in self.parameters():
            nn.init.uniform_(p, 0, 1 / sum(p.shape))

    def forward(self, input):
        x = input
        for module in self:
            z = module(x)
            h = functional.relu(z)
            x = torch.cat([h, input], dim=-1)
        return h

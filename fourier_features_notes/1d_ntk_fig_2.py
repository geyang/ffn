import jax
from jax import random, grad, jit, vmap
from jax.config import config
from jax.lib import xla_bridge
import jax.numpy as np
import neural_tangents as nt
from neural_tangents import stax
from jax.experimental import optimizers
import os

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from tqdm.notebook import tqdm as tqdm

import time

import numpy as onp

# Utils

fplot = lambda x : np.fft.fftshift(np.log10(np.abs(np.fft.fft(x))))

# Signal makers

def sample_random_signal(key, decay_vec):
    N = decay_vec.shape[0]
    raw = random.normal(key, [N, 2]) @ np.array([1, 1j])
    signal_f = raw * decay_vec
    signal = np.real(np.fft.ifft(signal_f))
    return signal

def sample_random_powerlaw(key, N, power):
    coords = np.float32(np.fft.ifftshift(1 + N//2 - np.abs(np.fft.fftshift(np.arange(N)) - N//2)))
    decay_vec = coords ** -power
    decay_vec = onp.array(decay_vec)
    decay_vec[N//4:] = 0
    return sample_random_signal(key, decay_vec)


# Network

def make_network(num_layers, num_channels, ntk_params=True, num_outputs=1):
    layers = []
    for i in range(num_layers-1):
        if ntk_params:
            layers.append(stax.Dense(num_channels, parameterization='standard'))
        else:
            layers.append(stax.Dense(num_channels, parameterization='standard'))
        layers.append(stax.Relu())
    layers.append(stax.Dense(num_outputs, parameterization='standard'))
    return stax.serial(*layers)

# Encoding

def compute_ntk(x, avals, bvals, kernel_fn):
    x1_enc = input_encoder(x, avals, bvals)
    x2_enc = input_encoder(np.array([0.], dtype=np.float32), avals, bvals)
    out = np.squeeze(kernel_fn(x1_enc, x2_enc, 'ntk'))
    return out


def input_encoder (x, a, b):
    return np.concatenate([a * np.sin((2.*np.pi*x[..., None]) * b),
                           a * np.cos((2.*np.pi*x[..., None]) * b)], axis=-1) / np.linalg.norm(a)


def predict_psnr_basic(kernel_fn, train_fx, test_fx, train_x, train_y, test_x, test_y, t_final, eta=None):
    g_dd = kernel_fn(train_x, train_x, 'ntk')
    g_td = kernel_fn(test_x, train_x, 'ntk')
    train_predict_fn = nt.predict.gradient_descent_mse(g_dd, train_y[...,None], g_td)
    print(t_final, train_fx.shape, test_fx.shape)
    train_theory_y, test_theory_y = train_predict_fn(t_final, train_fx[...,None], test_fx[...,None])

    calc_psnr = lambda f, g: -10. * np.log10(np.mean((f-g)**2))
    return calc_psnr(test_y, test_theory_y[:,0]), calc_psnr(train_y, train_theory_y[:,0])

# predict_psnr_basic = jit(predict_psnr_basic, static_argnums=(0,))


def train_model(rand_key, network_size, lr, iters,
                train_input, test_input, test_mask, optimizer, ab, name=''):
    if ab is None:
        ntk_params = False
    else:
        ntk_params = True
    init_fn, apply_fn, kernel_fn = make_network(*network_size, ntk_params=ntk_params)

    if ab is None:
        run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, x[...,None] - .5)))
    else:
        run_model = jit(lambda params, ab, x: np.squeeze(apply_fn(params, input_encoder(x, *ab))))
    model_loss = jit(lambda params, ab, x, y: .5 * np.sum((run_model(params, ab, x) - y) ** 2))
    model_psnr = jit(lambda params, ab, x, y: -10 * np.log10(np.mean((run_model(params, ab, x) - y) ** 2)))
    model_grad_loss = jit(lambda params, ab, x, y: jax.grad(model_loss)(params, ab, x, y))

    opt_init, opt_update, get_params = optimizer(lr)
    opt_update = jit(opt_update)

    if ab is None:
        _, params = init_fn(rand_key, (-1, 1))
    else:
        _, params = init_fn(rand_key, (-1, input_encoder(train_input[0], *ab).shape[-1]))
    opt_state = opt_init(params)

    pred0 = run_model(get_params(opt_state), ab, test_input[0])
    pred0_f = np.fft.fft(pred0)

    train_psnrs = []
    test_psnrs = []
    theories = []
    xs = []
    errs = []
    for i in tqdm(range(iters), desc=name):
        opt_state = opt_update(i, model_grad_loss(get_params(opt_state), ab, *train_input), opt_state)

        if i % 20 == 0:
            train_psnr = model_psnr(get_params(opt_state), ab, *train_input)
            test_psnr = model_psnr(get_params(opt_state), ab, test_input[0][test_mask], test_input[1][test_mask])
            if ab is None:
                train_fx = run_model(get_params(opt_state), ab, train_input[0])
                test_fx = run_model(get_params(opt_state), ab, test_input[0][test_mask])
                theory = predict_psnr_basic(kernel_fn, train_fx, test_fx, train_input[0][...,None]-.5, train_input[1], test_input[0][test_mask][...,None], test_input[1][test_mask], i*lr)
            else:
                test_x = input_encoder(test_input[0][test_mask], *ab)
                train_x = input_encoder(train_input[0], *ab)

                train_fx = run_model(get_params(opt_state), ab, train_input[0])
                test_fx = run_model(get_params(opt_state), ab, test_input[0][test_mask])
                theory = predict_psnr_basic(kernel_fn, train_fx, test_fx, train_x, train_input[1], test_x, test_input[1][test_mask], i*lr)


            train_psnrs.append(train_psnr)
            test_psnrs.append(test_psnr)
            theories.append(theory)
            pred = run_model(get_params(opt_state), ab, train_input[0])
            errs.append(pred - train_input[1])
            xs.append(i)
    return get_params(opt_state), train_psnrs, test_psnrs, errs, np.array(theories), xs

N_train = 32
data_power = 1

network_size = (4, 1024)

learning_rate = 1e-5
sgd_iters = 50001

rand_key = random.PRNGKey(0)

config.update('jax_disable_jit', False)

# Signal
M = 8
N = N_train
x_test = np.float32(np.linspace(0,1.,N*M,endpoint=False))
x_train = x_test[::M]

test_mask = onp.ones(len(x_test), onp.bool)
test_mask[np.arange(0,x_test.shape[0],M)] = 0

s = sample_random_powerlaw(rand_key, N*M, data_power)
s = (s-s.min()) / (s.max()-s.min()) - .5

# Kernels
bvals = np.float32(np.arange(1, N//2+1))
ab_dict = {}
# ab_dict = {r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0, 1]}
ab_dict = {r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0, 0.5, 1, 1.5, 2]}
ab_dict[r'$p = \infty$'] = (np.eye(bvals.shape[0])[0], bvals)
ab_dict['No mapping'] = None


# Train the networks

rand_key, *ensemble_key = random.split(rand_key, 1 + len(ab_dict))

outputs = {k : train_model(key, network_size, learning_rate, sgd_iters,
                           (x_train, s[::M]), (x_test, s), test_mask,
                           optimizer=optimizers.sgd, ab=ab_dict[k], name=k) for k, key in zip(ab_dict, ensemble_key)}

ab_dict.update({r'$p = {}$'.format(p) : (bvals**-np.float32(p), bvals) for p in [0.5, 1.5, 2]})
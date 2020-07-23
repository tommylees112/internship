import numpy as np
import filterpy.stats as stats
import matplotlib.pyplot as plt

from collections import namedtuple


# from Labbe et al.
# https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python
gaussian = namedtuple('Gaussian', ['mean', 'var'])
gaussian.__repr__ = lambda s: '𝒩(μ={:.3f}, 𝜎²={:.3f})'.format(s[0], s[1])


def update(prior, measurement):
    x, P = prior        # mean and variance of prior
    z, R = measurement  # mean and variance of measurement

    y = z - x        # residual
    K = P / (P + R)  # Kalman gain

    x = x + K*y      # posterior
    P = (1 - K) * P  # posterior variance
    return gaussian(x, P)


def predict(posterior, movement):
    x, P = posterior # mean and variance of posterior
    dx, Q = movement # mean and variance of movement
    x = x + dx
    P = P + Q
    return gaussian(x, P)


# Bayesian methodology
sensor_var = 1**2
process_var = 2**2
P0 = 500
pos = gaussian(noisy_data.mean(), P0)
process_model = gaussian(1., process_var)

zs, ps = [], []


# Non Bayesian methodology
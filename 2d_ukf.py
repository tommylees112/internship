# standard library
from pathlib import Path
import datetime
import random
import time
from typing import Dict, Tuple, Optional, Union, Any, List, Callable

# nearly standard library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Kalman Filter
from filterpy.common import Saver

# unscented
from filterpy.kalman import UKF
from filterpy.kalman import JulierSigmaPoints, MerweScaledSigmaPoints
from filterpy.kalman import unscented_transform

# custom libraries
from abc_model import PARAMETERS as parameters
from abc_model import abcmodel, abcmodel_matrix, abc_simulate, read_data
from utils import print_latex_matrices, update_data_columns
from abc_plots import plot_std_bands, plot_filtered_true_obs, plot_discharge_predictions


def abc(x: np.ndarray, params: Dict[str, float] = parameters) -> Tuple[float, float]:
    """function to propogate S (storage) forward in time
    producing a simulated discharge (q_sim)

    ::math
    q_t = (1 - a - b)r_t + cS_{t-1}
    S_t = (1 - c)S_{t - 1} + ar_t
    ::

    Args:
        S ([type]): [description]
        r ([type]): [description]
        parameters (Dict[str, float]): Dictionary with keys "a", "b", "c"

    Returns:
        storage [float]: propogate storage forward in time
    """
    a, b, c = params["a"], params["b"], params["c"]

    # q_sim = (1 - a - b) * r + (c * S)
    S = x[0]
    r = x[1]

    S = (1 - c) * S + (a * r)

    return S


def hx(z: np.ndarray, r: float, params: Dict[str, float] = parameters):
    """Create the measurement function to turn
    x into measurement space (z).
    Used to calculate the residual:

    y = z - Hx

    """
    a, b, c = params["a"], params["b"], params["c"]
    H = np.array([[c, (1 - a - b)], [0, 1]])

    return np.dot(H, z)


def init_filter(
    r0: float,
    S0: float = 5.74,
    S_var: float = 1,
    r_var: float = 100,
    Q00_s_noise: float = 0.01,
    Q11_r_noise: float = 10_000,
    R: float = 1,
    h: Callable = hx,
    f: Callable = abc,
    params: Dict = parameters,
):
    """Init the ABC model kalman filter

    X (state mean):
        [S0, r0]^T

    P (state uncertainty):
        [[S_var, 0    ]
         [0    , r_var]]

    ABC Model
    F (process transition matrix):
        [[1 - c,  a]
         [0    ,  1]]

    H (measurement function):
        [c, (1 - a - b)]^T

    Args:
        S_var (float, optional): [description]. Defaults to 1.
        r_var (float, optional): [description]. Defaults to 100.
        Q00_s_noise (float, optional): [description]. Defaults to 0.01.
        Q11_r_noise (float, optional): [description]. Defaults to 10_000.
        S0 (float, optional): [description]. Defaults to 5.74.

    Returns:
        [type]: [description]
    """
    assert all(np.isin(["a", "b", "c"], [k for k in params.keys()]))
    a, b, c = params["a"], params["b"], params["c"]

    # generate sigma points - van der Merwe method
    #   n = number of dimensions; alpha = how spread out;
    #   beta = prior knowledge about distribution, 2 means gaussian;
    #   kappa = scaling parameter, either 0 or 3-n;
    points = MerweScaledSigmaPoints(n=2, alpha=1e-3, beta=2, kappa=1)

    ## TODO: dt = 86400s (1day) (??)
    abc_filter = UKF.UnscentedKalmanFilter(
        dim_x=2, dim_z=2, dt=1, hx=hx, fx=abc, points=points
    )

    # INIT FILTER
    #  ------- Predict Variables -------
    # State Vector (X): storage and rainfall
    # (2, 1) = column vector
    abc_filter.x = np.array([[S0, r0]])

    # State Covariance (P) initial estimate
    # (2, 2) = square matrix
    abc_filter.P[:] = np.diag([S_var, r_var])

    # Process noise (Q)
    # (2, 2) = square matrix
    abc_filter.Q = np.diag([Q00_s_noise, Q11_r_noise])

    # ------- Update Variables -------
    # measurement uncertainty
    # (2, 2) = square matrix OR is it just uncertainty on discharge (q)
    abc_filter.R *= R

    # Control inputs (defaults)
    abc_filter.B = None  # np.ndarray([a])
    abc_filter.dim_u = 0

    return abc_filter


if __name__ == "__main__":
    base_dir = Path("/Users/tommylees/github/internship/")
    data_dir = base_dir / "data"
    plot_dir = base_dir / f"plots/UKF"

    if not plot_dir.exists():
        plot_dir.mkdir(exist_ok=True, parents=True)

    station_id = 39034

    # --- HYPERPARAMETERS --- #
    Q00_s_noise = 0.5
    Q11_r_noise = 10_000
    R = 0.01

    # --- DATA --- #
    original_data = read_data(data_dir)
    data = original_data.copy()
    data["q_prior"], data["S_prior"] = abc_simulate(data["r_obs"])

    # --- INIT FILTER --- #
    ukf = init_filter(
        r0=data.loc[0, "r_obs"],
        S0=5.74,
        S_var=1,
        r_var=1,
        Q00_s_noise=Q00_s_noise,
        Q11_r_noise=Q11_r_noise,
        R=R,
        h=hx,
        f=abc,
        params=parameters,
    )
    s = Saver(ukf)

    # --- RUN FILTER --- #
    uxs = []

    # TODO: can un-linearize the code and therefore have rainfall separate (?)
    for z, r in np.vstack([data["q_obs"], data["r_obs"]]).T:
        fx_args = hx_args = (r,)  #  rainfall inputs to the model
        # predict
        ukf.predict(*fx_args)

        # update
        ukf.update(np.array([z]), *hx_args)

        s.save()

    uxs = np.array(uxs)
    s.to_array()

    data = update_data_columns(data, s)

    # print_latex_matrices(s)

    # --- PLOTS --- #
    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=True)
    ax.set_ylim(-0.1, 4.5)
    plt.show()

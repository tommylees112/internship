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
from filterpy.kalman import unscented_transform as UT

# custom libraries
from abc_model import PARAMETERS as parameters
from abc_model import abcmodel, abcmodel_matrix, abc_simulate, read_data
from utils import print_latex_matrices, update_data_columns
from abc_plots import plot_std_bands, plot_filtered_true_obs, plot_discharge_predictions


def abc(
    x: np.ndarray, dt: int, r: float, params: Dict[str, float] = parameters
) -> Tuple[float, float]:
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
    # TODO: dt is currently not used
    a, b, c = params["a"], params["b"], params["c"]

    # q_sim = (1 - a - b) * r + (c * S)
    S = x[0]

    S = (1 - c) * S + (a * r)

    return np.array([S])


def hx(
    prior_sigma: np.ndarray, r: float, params: Dict[str, float] = parameters
) -> float:
    """Create the measurement function to turn
    prior_sigma (x) into measurement space (z).
    Used to calculate the residual:

    y = z - Hx
    y = z - h(X_i)

    """
    a, b, c = params["a"], params["b"], params["c"]

    q_sim = (1 - a - b) * r + (c * prior_sigma)

    return q_sim


def init_filter(
    S0: float = 5.74,
    S_var: float = 1,
    Q00_s_noise: float = 0.01,
    R: float = 1,
    h: Callable = hx,
    f: Callable = abc,
    params: Dict = parameters,
    alpha: float = 1e-3,
    beta: float = 2,
    kappa: float = 1,
) -> UKF.UnscentedKalmanFilter:
    """Init the ABC model kalman filter

    X (state mean):
        [S0]^T

    P (state uncertainty):
        [S_var]

    ABC Model
    F (process transition matrix):
    H (measurement function):

    Args:
        S0 (float, optional): [description]. Defaults to 5.74.
        S_var (float, optional): [description]. Defaults to 1.
        Q00_s_noise (float, optional): [description]. Defaults to 0.01.
        R (float, optional): [description]. Defaults to 1.
        h (Callable, optional): [description]. Defaults to hx.
        f (Callable, optional): [description]. Defaults to abc.
        params (Dict, optional): [description]. Defaults to parameters.
        alpha (float, optional): [description]. Defaults to 1e-3.
        beta (float, optional): [description]. Defaults to 2.
        kappa (float, optional): [description]. Defaults to 1.

    Returns:
        [UKF.UnscentedKalmanFilter]: Initialised Unscented Kalman Filter
    """
    assert all(np.isin(["a", "b", "c"], [k for k in params.keys()]))
    a, b, c = params["a"], params["b"], params["c"]

    # generate sigma points - van der Merwe method
    #   n = number of dimensions; alpha = how spread out;
    #   beta = prior knowledge about distribution, 2 == gaussian;
    #   kappa = scaling parameter, either 0 or 3-n;
    points = MerweScaledSigmaPoints(n=1, alpha=alpha, beta=beta, kappa=kappa)

    ## TODO: dt = 86400s (1day) (??)
    abc_filter = UKF.UnscentedKalmanFilter(
        dim_x=1, dim_z=1, dt=1, hx=h, fx=f, points=points
    )

    # INIT FILTER
    #  ------- Predict Variables -------
    # State Vector (X): storage and rainfall
    # (2, 1) = column vector
    abc_filter.x = np.array([S0])

    # State Covariance (P) initial estimate
    # (2, 2) = square matrix
    abc_filter.P[:] = np.diag([S_var])

    # Process noise (Q)
    # (2, 2) = square matrix
    abc_filter.Q = np.diag([Q00_s_noise])

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
    Q00_s_noise = 10  #  10
    R = 1e-2  #  1e-2
    alpha = 1e-3
    beta = 2
    kappa = 1

    # --- DATA --- #
    original_data = read_data(data_dir)
    data = original_data.copy()
    data["q_prior"], data["S_prior"] = abc_simulate(data["r_obs"])

    # --- INIT FILTER --- #
    ukf = init_filter(
        S0=5.74,
        S_var=1,
        Q00_s_noise=Q00_s_noise,
        R=R,
        h=hx,
        f=abc,
        params=parameters,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
    )
    s = Saver(ukf)

    # --- RUN FILTER --- #
    # TODO: how include measurement accuracy of rainfall ?
    for z, r in np.vstack([data["q_obs"], data["r_obs"]]).T:
        fx_args = hx_args = {"r": r}  #  rainfall inputs to the model

        # predict
        ukf.predict(**fx_args)

        # update
        ukf.update(z, **hx_args)

        s.save()

    # save the output data
    s.to_array()
    data = update_data_columns(data, s)

    print_latex_matrices(s)

    # --- PLOTS --- #
    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=False)
    ax.set_title(f"Predicted Discharge\nR: {R} Q: {Q00_s_noise}")
    ax.set_ylim(-0.1, 4.5)
    plt.show()
    fig.savefig(plot_dir / f"001_discharge_preds_{int( random.random() * 100 )}")

    fig, ax = plt.subplots()
    data.plot.scatter("q_obs", "q_filtered", c="q_variance", colormap="viridis", ax=ax)
    ax.set_title(f"Comparison with Observed Discharge\nR: {R} Q: {Q00_s_noise}")
    ax.set_xlabel("$q_{obs}$ [$mm day^{-1} km^{-2}$]")
    ax.set_ylabel("$q_{filtered}$ [$mm day^{-1} km^{-2}$]")
    sns.despine()
    fig.savefig(plot_dir / f"002_obs_filtered_scatter_{int( random.random() * 100 )}")

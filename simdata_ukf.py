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
from utils import print_latex_matrices, update_data_columns, calculate_r2_metrics
from abc_plots import (
    plot_std_bands,
    plot_filtered_true_obs,
    plot_discharge_predictions,
    plot_1_1_line,
)
from abc_simulation import ABCSimulation, plot_experiment_simulation


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


def init_2D_filter(
    r0: float,
    S_var: float = 1,
    r_variance: float = 100,
    Q00_s_noise: float = 0.01,
    Q11_r_noise: float = 10_000,
    S0: float = 5.74,
    R: float = 1,
    params: Dict = parameters,
    alpha: float = 1e-3,
    beta: float = 2,
    kappa: float = 1,
) -> UKF.UnscentedKalmanFilter:
    """Init the ABC model kalman filter

    X (state mean):
        [S, r]^T
    P (state uncertainty):
        [[S_var, 0    ]
         [0    , r_var]]

    ABC Model
    F (process transition matrix):
        [[1 - c,  a]
         [0    ,  1]]

    H (measurement function):
        [c, (1 - a - b)]^T

    Returns:
        UKF.UnscentedKalmanFilter: 2D UKF
    """
    assert all(np.isin(["a", "b", "c"], [k for k in params.keys()]))
    a, b, c = params["a"], params["b"], params["c"]

    # generate sigma points - van der Merwe method
    #   n = number of dimensions; alpha = how spread out;
    #   beta = prior knowledge about distribution, 2 == gaussian;
    #   kappa = scaling parameter, either 0 or 3-n;
    points = MerweScaledSigmaPoints(n=2, alpha=alpha, beta=beta, kappa=kappa)

    def fx2d(x, dt):
        a, b, c = params["a"], params["b"], params["c"]
        F = np.array([[1 - c, a], [0.0, 1.0]])
        x = np.dot(F, x)
        return x

    def hx2d(prior_sigma, P_matrix=False):
        a, b, c = params["a"], params["b"], params["c"]
        H = np.array([[c, (1 - a - b)], [0, 1]])
        if P_matrix:
            z_sigma = (H @ prior_sigma) @ np.transpose(H) # np.dot(np.dot(H, prior_sigma), np.transpose(H))
        else:  #   default
            z_sigma = np.dot(H, prior_sigma)
        return z_sigma

    ## TODO: dt = 86400s (1day) (??)
    abc_filter = UKF.UnscentedKalmanFilter(
        dim_x=2, dim_z=2, dt=1, hx=hx2d, fx=fx2d, points=points
    )

    # INIT FILTER
    #  ------- Predict Variables -------
    # State Vector (X): storage and rainfall
    # (2, 1) = column vector
    abc_filter.x = np.array([S0, r0]).T

    # State Covariance (P) initial estimate
    # (2, 2) = square matrix
    abc_filter.P[:] = np.diag([S_var, r_variance])

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
        S_var (float, optional): P_t=0. Defaults to 1.
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
    DIMENSION = 2
    base_dir = Path("/Users/tommylees/github/internship/")
    data_dir = base_dir / "data"
    plot_dir = base_dir / f"plots/UKF_sim"

    if not plot_dir.exists():
        plot_dir.mkdir(exist_ok=True, parents=True)

    station_id = 39034

    # --- DATA HYPERPARAMETERS --- #
    std_q_obs = 0.3
    std_r_obs = 1.5
    std_abc = 0.01
    std_S0 = 0.01

    # --- UKF HYPERPARAMETERS --- #
    Q00_s_noise = 10  #  10
    R = 1e-2  #  1e-2
    alpha = 1e-3
    beta = 2
    kappa = 1
    s_variance = 10  #  P[0, 0]  10
    r_variance = 10  #  P[1, 1]  10

    # 2D parameters
    Q11_r_noise = 1e5  #  10_000

    # --- DATA --- #
    simulator = ABCSimulation(
        data_dir,
        std_q_obs=std_q_obs,
        std_r_obs=std_r_obs,
        std_abc=std_abc,
        std_S0=std_S0,
    )
    data = simulator.data
    a_est, b_est, c_est = simulator.a_est, simulator.b_est, simulator.c_est
    a_true, b_true, c_true = simulator.a_true, simulator.b_true, simulator.c_true
    S0_est = simulator.S0_est
    S0_true = simulator.S0_true

    # --- INIT FILTER --- #
    if DIMENSION == 1:
        ukf = init_filter(
            S0=S0_est,
            S_var=s_variance,
            Q00_s_noise=Q00_s_noise,
            R=R,
            h=hx,
            f=abc,
            params=dict(a=a_est, b=b_est, c=c_est),
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )
    elif DIMENSION == 2:
        ukf = init_2D_filter(
            S0=S0_est,
            r0=data["r_obs"][0],
            S_var=s_variance,
            r_variance=r_variance,
            Q00_s_noise=Q00_s_noise,
            Q11_r_noise=Q11_r_noise,
            R=R,
            params=dict(a=a_est, b=b_est, c=c_est),
            alpha=alpha,
            beta=beta,
            kappa=kappa,
        )

    s = Saver(ukf)

    # --- RUN FILTER --- #
    # TODO: how include measurement accuracy of rainfall ?
    for z, r in np.vstack([data["q_obs"], data["r_obs"]]).T:
        if DIMENSION == 1:
            fx_args = hx_args = {"r": r}  #  rainfall inputs to the model

            # predict
            ukf.predict(**fx_args)

            # update
            ukf.update(z, **hx_args)
        elif DIMENSION == 2:
            z2d = np.array([z, r])
            ukf.predict()
            ukf.update(z2d)

        else:
            assert False, "Have only implemented [1, 2] dimensions"

        s.save()

    # save the output data
    s.to_array()
    data = update_data_columns(data, s, dimension=DIMENSION)

    print_latex_matrices(s)

    s_ukf = s
    data_ukf = data
    params = dict(a=a_est, b=b_est, c=c_est)
    H = np.array([[params["c"], (1 - params["a"] - params["b"])], [0, 1]])

    # --- PLOTS --- #
    # fig, ax = plot_discharge_predictions(
    #     data, filtered_prior=False, plusR=False)
    # ax.set_ylim(-0.1, 4.5)
    # plt.show()

    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=False)
    ax.set_title(
        f"UNSCENTED KF ({DIMENSION}D) Predicted Discharge R: {R} Q: {Q00_s_noise} "
        "\n $\sigma_{q_{obs}}$:"
        f"{std_q_obs} "
        "$\sigma_{r_{obs}}$:"
        f"{std_r_obs} "
        "$\sigma_{abc}$:"
        f"{std_abc} "
        "$\sigma_{S0}$:"
        f"{std_S0}"
    )
    ax.set_ylim(-0.1, 4.5)
    plt.show()
    # fig.savefig(plot_dir / f"001_discharge_preds_{int( random.random() * 100 )}")

    fig, ax = plt.subplots()
    data.plot.scatter("q_obs", "q_filtered", c="q_variance", colormap="viridis", ax=ax)
    r2_df = calculate_r2_metrics(data)
    r2 = float(r2_df.loc[r2_df["run"] == "posterior", "r2"])
    ax.set_title(
        f"UKF 2D - $R^2$ Score: {r2:.2f} R: {R} Q: {Q00_s_noise}"
        "\n $\sigma_{q_{obs}}$:"
        f"{std_q_obs} "
        "$\sigma_{r_{obs}}$:"
        f"{std_r_obs} "
        "$\sigma_{abc}$:"
        f"{std_abc} "
        "$\sigma_{S0}$:"
        f"{std_S0}"
    )
    ax.set_xlabel("$q_{obs}$ [$mm day^{-1} km^{-2}$]")
    ax.set_ylabel("$q_{filtered}$ [$mm day^{-1} km^{-2}$]")
    ax = plot_1_1_line(ax)
    ax.legend()
    sns.despine()
    plt.show()
    # fig.savefig(plot_dir / f"002_obs_filtered_scatter_{int( random.random() * 100 )}")

    # Plot the Simulated Data
    # plot_experiment_simulation(data)

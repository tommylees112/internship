import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any, List
from pathlib import Path
from collections import Iterable
import random
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import filterpy.kalman as kf
from filterpy.kalman import KalmanFilter
from filterpy.common import Saver
from filterpy.stats import plot_covariance_ellipse, plot_covariance

from abc_model import abcmodel, abcmodel_matrix, abc_simulate
from abc_model import PARAMETERS as parameters
from abc_plots import (
    plot_simulated_data,
    plot_qprior_predictions,
    plot_predicted_observed_discharge,
    plot_simulated_discharge,
    plot_discharge_predictions,
    plot_state_storage,
    plot_discharge_uncertainty,
    plot_possible_draws,
    plot_prior_posterior_scatter_r2,
    plot_filtered_true_obs,
    setup_discharge_plot,
    plot_residual_limits,
    plot_uncertainties,
)
from config import read_config
from abc_simulation import ABCSimulation
from utils import print_latex_matrices, update_data_columns, calculate_r2_metrics


# Set seeds (for reproducibility)
random.seed(1)
np.random.seed(1)


# --------------- IO FUNCTIONS -----------------
def create_rundir(run_dir: Path, experiment_name: str) -> Path:
    # get the time now
    now = datetime.datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)

    # create the run name
    folder = f"{experiment_name}_{month}{day}:{hour}{minute}"

    # create the folder and parents
    (run_dir / folder).mkdir(exist_ok=True, parents=True)
    (run_dir / folder / "plots").mkdir(exist_ok=True, parents=True)

    return run_dir / folder


# --------------- KF FUNCTIONS -----------------
def init_filter(
    r0: float,
    s_variance: float = 1,
    r_variance: float = 100,
    Q00_s_noise: float = 0.01,
    Q11_r_noise: float = 10_000,
    S0: float = 5.74,
    R: float = 1,
    params: Dict = parameters,
):
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

    Args:
        s_variance (float, optional): [description]. Defaults to 1.
        r_variance (float, optional): [description]. Defaults to 100.
        Q00_s_noise (float, optional): [description]. Defaults to 0.01.
        Q11_r_noise (float, optional): [description]. Defaults to 10_000.
        S0 (float, optional): [description]. Defaults to 5.74.

    Returns:
        [type]: [description]
    """
    assert all(np.isin(["a", "b", "c"], [k for k in params.keys()]))
    a, b, c = params["a"], params["b"], params["c"]
    abc_filter = KalmanFilter(dim_x=2, dim_z=2, dim_u=0)

    # INIT FILTER
    #  ------- Predict Variables -------
    # State Vector (X): storage and rainfall
    # (2, 1) = column vector
    abc_filter.x = np.array([[S0, r0]]).T

    # State Covariance (P) initial estimate
    # (2, 2) = square matrix
    abc_filter.P[:] = np.diag([s_variance, r_variance])

    #  state transition (F) - the process model
    # (2, 2) = square matrix
    abc_filter.F = np.array([[1 - c, a], [0.0, 1.0]])

    # Process noise (Q)
    # (2, 2) = square matrix
    abc_filter.Q = np.diag([Q00_s_noise, Q11_r_noise])

    # ------- Update Variables -------
    # Measurement function (H) (how do we go from state -> observed?)
    # (1, 2) = row vector
    abc_filter.H = np.array([[c, (1 - a - b)], [0, 1]])

    # measurement uncertainty
    # (2, 2) = square matrix OR is it just uncertainty on discharge (q)
    abc_filter.R *= R

    # Control inputs (defaults)
    abc_filter.B = None  # np.ndarray([a])
    abc_filter.dim_u = 0

    return abc_filter


def run_kf(
    data: pd.DataFrame,
    s_variance: float = 1,
    r_variance: float = 100,
    Q00_s_noise: float = 0.1,
    Q11_r_noise: float = 10_000,
    R: float = 0.01,
    S0: float = 5.74,
    observe_every: int = 1,
    params: dict = parameters,
) -> Tuple[KalmanFilter, Saver, pd.DataFrame]:

    assert all(np.isin(["q_obs", "r_obs"], data.columns))
    # ------ INIT FILTER ------
    kf = init_filter(
        r0=data["r_obs"][0],
        S0=S0,
        s_variance=s_variance,
        r_variance=r_variance,
        Q00_s_noise=Q00_s_noise,
        Q11_r_noise=Q11_r_noise,
        R=R,
        params=params,
    )

    s = Saver(kf)

    # ------ RUN FILTER ------
    if observe_every > 1:
        print(f"Running with observe_every parameter == {observe_every}")
        data.loc[data.index % observe_every != 0, "q_obs"] = np.nan

    # Iterate over the Kalman Filter
    for time_ix, z in enumerate(np.vstack([data["q_obs"], data["r_obs"]]).T):
        kf.predict()
        # only make update steps every n timesteps
        if time_ix % observe_every == 0:
            kf.update(z)
        s.save()

    s.to_array()

    data = update_data_columns(data, s, dimension=None)

    # # only observe every n values
    # # data["q_true_original"] = data["q_true"]

    # # update data with POSTERIOR estimates
    # # Calculate the DISCHARGE (measurement operator * \bar{x})
    # data["q_filtered"] = ((s.H @ s.x))[:, 0]
    # data["q_variance"] = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)))[:, 0, 0]
    # data["q_variance_plusR"] = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)) + s.R)[
    #     :, 0, 0
    # ]

    # data["s_variance"] = s.P[:, 0, 0]
    # data["s_variance_plusR"] = (s.P + s.R)[:, 0, 0]
    # data["s_filtered"] = s.x[:, 0]

    # data["q_prior2"] = ((s.H @ s.x_prior))[:, 0]

    return kf, s, data


def calculate_r2_metrics(data):
    data = data.dropna()
    # unfiltered prediction
    prior_r2 = r2_score(data["q_true"], data["q_prior"])
    #  filtered prior prediction
    if "q_x_prior" in data.columns:
        filtered_prior_r2 = r2_score(data["q_true"], data["q_x_prior"])

    # filtered posterior prediction
    posterior_r2 = r2_score(data["q_true"], data["q_filtered"])

    r2 = pd.DataFrame(
        {
            "run": ["posterior", "prior", "filtered_prior"],
            "r2": [posterior_r2, prior_r2, filtered_prior_r2],
        }
    )
    return r2


# --------------- MAIN CODE -----------------
if __name__ == "__main__":
    # ------ HYPER PARAMS ------
    # data simulation params
    std_q_obs = 0.3
    std_r_obs = 1.5
    std_abc = 0.01
    std_S0 = 0.01

    #  kalman filter params
    # MEASUREMENT
    R = 0.01  # 0.01
    # PROCESS
    S0 = initial_state = 5.74  # 5.74
    s_variance = 10  #  P[0, 0]  10
    r_variance = 10  #  P[1, 1]  10
    Q00_s_noise = 1  #  Q[0, 0] 10  0.1
    Q11_r_noise = 1e5  #  Q[1, 1] 10_000

    # How often to make observations?
    observe_every = 1
    assert observe_every >= 1, "Expect observe_every to be at least one"

    # ------ SETUP RUN ------
    base_dir = Path("/Users/tommylees/github/internship/")
    data_dir = base_dir / "data"
    plot_dir = base_dir / f"plots/simdata_LKF"

    if not plot_dir.exists():
        plot_dir.mkdir(exist_ok=True, parents=True)

    station_id = 39034

    # ------ SIMULATE data ------
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

    # ------ RUN THE FILTER ------
    kf, s, data = run_kf(
        data=data,
        s_variance=s_variance,
        r_variance=r_variance,
        Q00_s_noise=Q00_s_noise,
        Q11_r_noise=Q11_r_noise,
        R=R,
        S0=S0_est,
        observe_every=1,
        params=dict(a=a_est, b=b_est, c=c_est)
    )

    print_latex_matrices(s)

    s_lkf = s
    data_lkf = data
    lkf = kf

    # ------ INTERPRET OUTPUT ------
    # 2. prior, filtered, true discharge Lineplot
    # fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=True)
    # ax.set_ylim(-0.1, 4.5)
    # plt.show()
    # fig.savefig(plot_dir / "005_prior_true_sim_discharge_plusR.png")
    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=False)
    ax.set_title(
        f"LINEAR KF Predicted Discharge R: {R} Q: {Q00_s_noise} "
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
    # fig.savefig(plot_dir / "005_prior_true_sim_discharge.png")

    # 3. Plot the Storage Parameter (unobserved)
    # fig, ax = plot_state_storage(s, data, plusR=True)
    # plt.show()
    # fig.savefig(plot_dir / "006_prior_true_sim_storage_plusR.png")

    # fig, ax = plot_state_storage(s, data, plusR=False)
    # plt.show()
    # fig.savefig(plot_dir / "006_prior_true_sim_storage.png")

    # fig, ax = plot_discharge_uncertainty(data)
    # plt.show()
    # fig.savefig(plot_dir / "007_discharge_uncertainty.png")

    # #
    # fig, ax = plt.subplots(figsize=(12, 4))
    # ax = plot_filtered_true_obs(data)
    # ax.set_title("The Kalman Filter Posterior")
    # fig.savefig(plot_dir / "008_true_obs.png")

    # fig, ax = plot_raw_discharge()
    # fig.savefig(plot_dir / "RAW discharge.png")

    # plt.close("all")


# means = s.x[:, 0, 0]
# variances = s.P[:, 0, 0]
# stds = 3
# fig, ax = plot_uncertainties(means, variances, stds)
# ax.set_title("$\sigma^2 = P$ for $S_t$")

# means = (s.H @ s.x)[:, 0, 0]
# variances = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)))[:, 0, 0]
# stds = 3
# fig, ax = plot_uncertainties(means, variances, stds)
# ax.set_title("$\sigma^2 = HPH^T$ for $q_t$")

# means = (s.H @ s.y)[:, 0, 0]
# variances = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)))[:, 0, 0]
# stds = 1
# fig, ax = plot_uncertainties(means, variances, stds)
# ax.set_title("$\mu = Hy_t$ $\sigma^2 = HPH^T$ for $q_t$")

# means = (s.H @ s.y)[:, 0, 0]
# variances = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)))[:, 0, 0]
# stds = 1
# fig, ax = plot_uncertainties(means, variances, stds)
# ax.set_title("$\mu = Hy_t$ $\sigma^2 = HPH^T$ for $q_t$")

# means = (s.y)[:, 1, 0]
# variances = (s.P)[:, 1, 1]
# stds = 1
# fig, ax = plot_uncertainties(means, variances, stds)
# ax.set_title("$\mu = Hy_t$ $\sigma^2 = HPH^T$ for $q_t$")

# fig, ax = plt.subplots()
# pred_r = s.x[:, 1, 0]
# ax.plot(data.index, pred_r, label="Predicted Rainfall (x)")
# obs_r = data["precipitation"]
# ax.scatter(
#     data.index,
#     pred_r,
#     label="Obs/input Rainfall (z)",
#     marker="x",
#     color=sns.color_palette()[1],
# )

# fig, ax = plt.subplots()
# # rainfall residuals
# ys = (s.z - (s.H @ s.x_prior))[:, 1, 0]
# sys = s.y[:, 1, 0]
# ax.scatter(data.index, ys, label="Calculated residuals")
# ax.scatter(data.index, s.y[:, 1, 0], label="Saver residuals", marker="x")
# plt.legend()
# sns.despine()

# #  Plot covariance ellipses!
# fig, ax = plt.subplots()
# plot_covariance(kf.x, kf.P, [1, 2, 3])
# ax.set_xlabel("Uncertainty in Storage (P[0,0])")
# ax.set_ylabel("Uncertainty in Rainfall (P[1,1])")
# sns.despine()

# fig, ax = plt.subplots()
# plot_covariance(kf.x, kf.R, [1, 2, 3])
# ax.set_xlabel("Measurement uncertainty in Discharge (R[0,0])")
# ax.set_ylabel("Measurement uncertainty in Rainfall (R[1,1])")
# sns.despine()

# fig, ax = plt.subplots()
# plot_covariance(kf.x, kf.Q, [1, 2, 3])
# ax.set_xlabel("Measurement uncertainty in Storage (Q[0,0])")
# ax.set_ylabel("Measurement uncertainty in Rainfall (Q[1,1])")
# sns.despine()

# fig, ax = plt.subplots()
# ax.plot(data.index, s.log_likelihood)
# sns.despine()


def plot_rainfall_transitions(s: Saver):
    fig, ax = plt.subplots(figsize=(12, 4))
    rain_obs = s.z[:, 1].flatten()
    rain_state = s.x[:, 1].flatten()
    rain_prior = s.x_prior[:, 1].flatten()
    rain_uncertainty = s.P_post[:, 1, 1].flatten()
    ax.plot(np.arange(len(rain_obs)), rain_obs, label="Observed Rainfall")
    ax.scatter(
        np.arange(len(rain_state)),
        rain_state,
        label="State Rainfall - $x_{posterior}$",
        marker="x",
    )
    ax.scatter(
        np.arange(len(rain_prior)),
        rain_prior,
        label="Prior Rainfall - $x_{prior}$ ($F = I$)",
        marker="o",
        facecolor="none",
        edgecolor="k",
    )
    ax.fill_between(
        np.arange(len(rain_state)),
        rain_obs - np.sqrt(rain_uncertainty),
        rain_obs + np.sqrt(rain_uncertainty),
        label="$+- \sigma_{r}$",
        alpha=0.3,
        color=sns.color_palette()[0],
    )
    ax.set_title("Measured, Mean and Covariance of Precipitation")
    ax.legend()
    sns.despine()

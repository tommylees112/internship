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


# Set seeds (for reproducibility)
random.seed(1)
np.random.seed(1)


# --------------- IO FUNCTIONS -----------------
def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(data_dir / "39034_2010.csv")


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
    s_noise: float = 0.01,
    r_noise: float = 10_000,
    S0: float = 5.74,
    R: float = 1,
    parameters: Dict = parameters,
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
        s_noise (float, optional): [description]. Defaults to 0.01.
        r_noise (float, optional): [description]. Defaults to 10_000.
        S0 (float, optional): [description]. Defaults to 5.74.

    Returns:
        [type]: [description]
    """
    assert all(np.isin(["a", "b", "c"], [k for k in parameters.keys()]))
    a, b, c = parameters["a"], parameters["b"], parameters["c"]
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
    abc_filter.Q = np.diag([s_noise, r_noise])

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


def simulate_data(
    original_data: pd.DataFrame, q_obs_noise: float, r_obs_noise: float
) -> pd.DataFrame:
    assert all(np.isin(["precipitation", "discharge_spec"], original_data.columns))
    data = original_data.copy()

    # simulate using the ABC model
    true_q, true_S = abc_simulate(data["precipitation"])
    data["q_true"] = true_q
    data["S_true"] = true_S

    # create q_obs (add noise)
    q_noise = np.random.normal(0, np.sqrt(q_obs_noise), len(data["discharge_spec"]))
    data["q_obs"] = np.clip(data["q_true"] + q_noise, a_min=0, a_max=None)

    # create r_obs
    r_noise = np.random.normal(0, np.sqrt(r_obs_noise), len(data["discharge_spec"]))
    data["r_obs"] = np.clip(data["precipitation"] + r_noise, a_min=0, a_max=None)

    # create S_prior, q_prior (without KF)
    prior_q, prior_S = abc_simulate(data["r_obs"])
    data["q_prior"] = prior_q
    data["S_prior"] = prior_S

    return data


def run_kf(
    data: pd.DataFrame,
    s_variance: float = 1,
    r_variance: float = 3,
    s_noise: float = 0.1,
    r_noise: float = 10_000,
    R: float = 0.01,
    S0: float = 5.74,
    observe_every: int = 1,
) -> Tuple[KalmanFilter, Saver, pd.DataFrame]:

    assert all(np.isin(["q_obs", "r_obs"], data.columns))
    # ------ INIT FILTER ------
    kf = init_filter(
        r0=data["r_obs"][0],
        S0=S0,
        s_variance=s_variance,
        r_variance=r_variance,
        s_noise=s_noise,
        r_noise=r_noise,
        R=R,
    )

    s = Saver(kf)

    # ------ RUN FILTER ------
    if observe_every > 1:
        data.loc[data.index % observe_every != 0, "q_obs"] = np.nan

    # Iterate over the Kalman Filter
    for time_ix, z in enumerate(np.vstack([data["q_obs"], data["r_obs"]]).T):
        kf.predict()
        # only make update steps every n timesteps
        if time_ix % observe_every == 0:
            kf.update(z)
        s.save()

    s.to_array()

    # only observe every n values
    # data["q_true_original"] = data["q_true"]

    # update data with POSTERIOR estimates
    # Calculate the DISCHARGE (measurement operator * \bar{x})
    data["q_filtered"] = ((s.H @ s.x))[:, 0]
    data["q_variance"] = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)))[:, 0, 0]
    data["q_variance_plusR"] = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)) + s.R)[
        :, 0, 0
    ]

    data["s_variance"] = s.P[:, 0, 0]
    data["s_variance_plusR"] = (s.P + s.R)[:, 0, 0]
    data["s_filtered"] = s.x[:, 0]

    data["q_prior2"] = ((s.H @ s.x_prior))[:, 0]

    return kf, s, data


def print_latex_matrices(s: Saver):
    Q = s.Q[0, :, :]
    R = s.R[0, :, :]
    print(
        "Q=\\left[\\begin{array}{cc}"
        f"{Q[0, 0]} & 0 \\\ "
        f"0 & {Q[1, 1]}"
        "\\end{array}\\right]"
    )
    print(
        "R=\\left[\\begin{array}{cc}"
        f"{R[0, 0]} & 0 \\\ "
        f"0 & {R[1, 1]}"
        "\\end{array}\\right]"
    )


def calculate_r2_metrics(data):
    data = data.dropna()
    prior_r2 = r2_score(data["q_true"], data["q_prior"])
    posterior_r2 = r2_score(data["q_true"], data["q_filtered"])
    r2 = pd.DataFrame({"run": ["posterior", "prior"], "r2": [posterior_r2, prior_r2]})
    return r2


# --------------- MAIN CODE -----------------
if __name__ == "__main__":
    # ------ HYPER PARAMS ------
    # data simulation params
    # r_obs_noise = 3.0           # 3.0
    # q_obs_noise = 0.01          # 0.01

    #  kalman filter params
    # MEASUREMENT
    R = 0.01  # 0.01
    # PROCESS
    S0 = initial_state = 5.74  # 5.74
    s_variance = 10  #  P[0, 0]  10
    r_variance = 10  #  P[1, 1]  10
    s_noise = 1e5  #  Q[0, 0] 10
    r_noise = 1e5  #  Q[1, 1] 10_000

    # How often to make observations?
    observe_every = 1
    assert observe_every >= 1, "Expect observe_every to be at least one"

    # ------ SETUP RUN ------
    base_dir = Path("/Users/tommylees/github/internship/")
    data_dir = base_dir / "data"
    plot_dir = base_dir / f"plots/Q00[{s_noise}]_Q11[{r_noise}]_R[{R}]"

    if not plot_dir.exists():
        plot_dir.mkdir(exist_ok=True, parents=True)

    station_id = 39034
    original_data = read_data(data_dir)

    # ------ Get data from CAMELS ------
    data = original_data.copy()
    data["q_obs"] = data["discharge_spec"]
    data["r_obs"] = data["precipitation"]
    data["q_prior"], data["S_prior"] = abc_simulate(data["r_obs"])

    # ------ RUN THE FILTER ------
    kf, s, data = run_kf(
        data=data,
        s_variance=s_variance,
        r_variance=r_variance,
        s_noise=s_noise,
        r_noise=r_noise,
        R=R,
        S0=S0,
        observe_every=observe_every,
    )

    print_latex_matrices(s)

    # ------ INTERPRET OUTPUT ------
    # 2. prior, filtered, true discharge Lineplot
    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=True)
    ax.set_ylim(-0.1, 4.5)
    # plt.show()
    fig.savefig(plot_dir / "005_prior_true_sim_discharge_plusR.png")
    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=False)
    ax.set_ylim(-0.1, 4.5)
    # plt.show()
    fig.savefig(plot_dir / "005_prior_true_sim_discharge.png")

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

    plt.close("all")

    d = data.copy()
    d = d.rename(dict(
        precipitation='r',
        discharge_spec='y',
        S_prior='S',
        q_prior="y_hat",
    ), axis=1)
    d["y_t-1"] = d["y"].shift(1)
    d = d[["time", "y", "r", "S", "y_hat", "y_t-1"]]
    d["target"] = d["y"] - d["y_hat"]
    d["input"] = tuple(zip(d["y_t-1"], d["y_hat"], d["r"]))



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

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
)
from config import read_config


# --------------- IO FUNCTIONS -----------------


def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(data_dir / "39034_2010.csv")


def create_rundir(run_dir: Path, experiment_name: str) -> Path:
    # get the time now
    now = datetime.now()
    day = f"{now.day}".zfill(2)
    month = f"{now.month}".zfill(2)
    hour = f"{now.hour}".zfill(2)
    minute = f"{now.minute}".zfill(2)

    # create the run name
    folder = f"{experiment_name}_{month}{day}:{hour}{minute}"

    # create the folder and parents
    (run_dir / folder).mkdir(exist_ok=True, parents=True)

    return run_dir / folder


# --------------- KF FUNCTIONS -----------------
def init_filter(
    r0: float,
    s_uncertainty: float = 1,
    r_uncertainty: float = 100,
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
        s_uncertainty (float, optional): [description]. Defaults to 1.
        r_uncertainty (float, optional): [description]. Defaults to 100.
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
    abc_filter.P[:] = np.diag([s_uncertainty, r_uncertainty])

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


# --------------- MAIN CODE -----------------

if __name__ == "__main__":
    # ------ HYPER PARAMS ------
    # data simulation params
    S0 = initial_state = 5.74
    r_obs_noise = 3.0
    q_obs_noise = 0.01

    #  kalman filter params
    R = 0.01                # q_obs_noise
    s_uncertainty = 1       # P[0, 0]
    r_uncertainty = 100     # P[1, 1]
    s_noise = 0.1
    r_noise = 10_000

    # How often to make observations?
    observe_every = 1
    assert observe_every >= 1, "Expect observe_every to be at least one"

    # ------ SETUP RUN ------
    base_dir = Path("/Users/tommylees/github/internship/")
    data_dir = base_dir / "data"
    plot_dir = base_dir / "plots"

    station_id = 39034
    original_data = read_data(data_dir)

    # ------ SIMULATE DATA ------
    data = simulate_data(
        original_data=original_data, q_obs_noise=q_obs_noise, r_obs_noise=r_obs_noise
    )

    # q_true vs. q_obs
    fig, ax = plot_simulated_data(data["q_true"], data["q_obs"])
    plt.show()
    fig.savefig(plot_dir / "001_qobs_qtrue.png")

    # r_true vs. r_obs
    fig, ax = plot_simulated_data(data["precipitation"], data["r_obs"])
    ax.set_ylabel("Precipitation ($r$ - $mm day^{-1}$")
    ax.set_title("$r_{obs}$ vs. $r_{true}$")
    plt.show()
    fig.savefig(plot_dir / f"002_robs_rtrue.png")

    # plot the PRIOR predictions
    fig, ax = plot_qprior_predictions(data)
    plt.show()
    fig.savefig(plot_dir / f"003_qprior.png")


    # ------ INIT FILTER ------
    kf = init_filter(
        r0=data["r_obs"][0],
        S0=S0,
        s_uncertainty=s_uncertainty,
        r_uncertainty=r_uncertainty,
        s_noise=s_noise,
        r_noise=r_noise,
        R=R,
    )

    s = Saver(kf)

    # ------ RUN FILTER ------
    # Iterate over the Kalman Filter
    # for z, u in zip(data["q_obs"], data["precipitation"]):
    for time_ix, z in enumerate(np.vstack([data["q_obs"], data["r_obs"]]).T):
        kf.predict()
        # only make update steps every n timesteps
        if time_ix % observe_every == 0:
            kf.update(z)
        s.save()

    s.to_array()

    if observe_every > 1:
        data.loc[data.index % observe_every != 0, "q_true"] = np.nan

    # Calculate the DISCHARGE (measurement operator * \bar{x})
    data["q_filtered"] = (s.H @ s.x)[:, 0]
    data["q_variance"] = (s.H @ s.P)[:, 0, 0]

    # ------ INTERPRET OUTPUT ------
    # calculate error metrics
    prior_r2 = r2_score(data["q_true"], data["q_prior"])
    posterior_r2 = r2_score(data["q_true"], data["q_filtered"])
    print("R2 Metrics:")
    print(f"Prior R2: {prior_r2:.2f}")
    print(f"Posterior R2: {posterior_r2:.2f}")

    # 1. Filtered/Prior vs. True Scatter
    fig, axs = plt.subplots(1, 2, figsize=(6 * 2, 4))
    plot_predicted_observed_discharge(data, s, ax=axs[0])
    plot_simulated_discharge(data, ax=axs[1])
    plt.show()
    fig.savefig(plot_dir / "004_discharge_scatter.png")

    # 2. prior, filtered, true discharge Lineplot
    fig, ax = plot_discharge_predictions(data)
    plt.show()
    fig.savefig(plot_dir / "005_prior_true_sim_discharge.png")

    # 3. Plot the Storage Parameter (unobserved)
    plot_state_storage(s, data)
    plt.show()
    fig.savefig(plot_dir / "006_prior_true_sim_storage.png")

    fig, ax = plot_discharge_uncertainty(data)
    plt.show()
    fig.savefig(plot_dir / "007_discharge_uncertainty.png")


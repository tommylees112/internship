import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any, List
from pathlib import Path

import random

import seaborn as sns
import matplotlib.pyplot as plt

from filterpy.kalman import KalmanFilter
from filterpy.common import Saver

from abc_model import PARAMETERS as parameters
from abc_plots import plot_discharge_predictions
from abc_simulation import ABCSimulation
from utils import print_latex_matrices, update_data_columns, calculate_r2_metrics
from simdata_lkf import run_kf
from simdata_ukf import run_ukf


if __name__ == "__main__":
    ### EXPERIMENT CHANGES ### 
    UKF_DIMENSION = 2
    observe_every = 1
    plot_discharge = True
    Q00_s_noise = 0.01  #  Q[0, 0] 10  0.1
    Q11_r_noise = 1e4  #  Q[1, 1] 10_000

    # --- INIT --- #
    base_dir = Path("/Users/tommylees/github/internship/")
    data_dir = base_dir / "data"
    plot_dir = base_dir / f"plots/UKF_sim"

    if not plot_dir.exists():
        plot_dir.mkdir(exist_ok=True, parents=True)

    station_id = 39034

    # --- SIMDATA HYPERPARAMETERS --- #
    std_q_obs = 0.3
    std_r_obs = 1.5
    std_abc = 0.01
    std_S0 = 0.01

    # --- KF HYPERPARAMETERS --- #
    R = 1e-2  #  1e-2
    alpha = 1e-3  #         1e-3
    beta = 2  #             2
    kappa = 1  #            1
    s_variance = 10  #  P[0, 0]  10
    r_variance = 10  #  P[1, 1]  10

    print(f"OBSERVE EVERY: {observe_every}")
    print(f"UKF_DIMENSION: {UKF_DIMENSION}")
    print(f"Storage Process Noise (Q[0, 0]): {Q00_s_noise}")
    print(f"Rainfall Process Noise (Q[1, 1]): {Q11_r_noise}")

    # --- DATA --- #
    simulator = ABCSimulation(
        data_dir,
        log_precip=True,
        std_q_obs=std_q_obs,
        std_r_obs=std_r_obs,
        std_abc=std_abc,
        std_S0=std_S0,
    )
    data = simulator.data.copy()
    a_est, b_est, c_est = simulator.a_est, simulator.b_est, simulator.c_est
    a_true, b_true, c_true = simulator.a_true, simulator.b_true, simulator.c_true
    S0_est = simulator.S0_est
    S0_true = simulator.S0_true

    params = dict(a=a_est, b=b_est, c=c_est)

    # ------ RUN THE LINEAR FILTER ------
    lkf, s_lkf, data_lkf = run_kf(
        data=data,
        s_variance=s_variance,
        r_variance=r_variance,
        Q00_s_noise=Q00_s_noise,
        Q11_r_noise=Q11_r_noise,
        R=R,
        S0=S0_est,
        observe_every=observe_every,
        params=params,
    )

    print("\n LINEAR KF:")
    print_latex_matrices(s_lkf)

    # --- RUN THE UNSCENTED KF --- #
    ukf, data_ukf, s_ukf = run_ukf(
        data=data,
        S0_est=S0_est,
        s_variance=s_variance,
        r_variance=r_variance,
        Q00_s_noise=Q00_s_noise,
        Q11_r_noise=Q11_r_noise,
        R=R,
        params=params,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        dimension=UKF_DIMENSION,
        observe_every=observe_every,
    )

    print("\n UNSCENTED KF:")
    print_latex_matrices(s_ukf)

    print(
        "\\\\""\\alpha = " f"{alpha}" "\\\\"
        "\\beta = "f"{beta}" "\\\\"
        "\\kappa = "f"{kappa}" "\\\\"
    )

    # --- PLOTS ---
    ylim = (-0.1, 2.5)
    # ukf
    fig, ax = plot_discharge_predictions(data_ukf, filtered_prior=False, plusR=True)
    ax.set_title(
        f"UNSCENTED KF ({UKF_DIMENSION}D) Predicted Discharge R: {R} Q: {Q00_s_noise} "
        "\n $\sigma_{q_{obs}}$:"
        f"{std_q_obs} "
        "$\sigma_{r_{obs}}$:"
        f"{std_r_obs} "
        "$\sigma_{abc}$:"
        f"{std_abc} "
        "$\sigma_{S0}$:"
        f"{std_S0}"
    )
    ax.set_ylim(ylim)
    ax.set_xlim(-1, 50)
    plt.show()
    fig.savefig(
        f"/Users/tommylees/Downloads/00_unscented_kf.png"
    )

    # lkf
    fig, ax = plot_discharge_predictions(data_lkf, filtered_prior=False, plusR=True)
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
    ax.set_ylim(ylim)
    ax.set_xlim(-1, 50)
    plt.show()
    fig.savefig(f"/Users/tommylees/Downloads/00_linear_kf.png")

    # plot the mean posterior prediction
    fig, axs = plt.subplots(2, 1, figsize=(12, 4 * 2))

    ax = axs[0]
    if plot_discharge:
        data_lkf["q_filtered"].plot(
            ax=ax, color=sns.color_palette()[0], label="Linear $q_{filtered}$"
        )
        data_ukf["q_filtered"].plot(
            ax=ax, color=sns.color_palette()[1], label="Unscented $q_{filtered}$"
        )
        ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")

    else:
        ax.plot(
            np.arange(len(s_lkf.P[:, 0, 0])),
            s_lkf.x[:, 0],
            c=sns.color_palette()[0],
            label="Linear $S_{filtered}$",
        )
        ax.plot(
            np.arange(len(s_ukf.P[:, 0, 0])),
            s_ukf.x[:, 0],
            c=sns.color_palette()[1],
            label="Unscented $S_{filtered}$",
        )
        ax.set_ylabel("Storage ($S$)")
    ax.set_title("Mean Predictions")
    ax.set_xlabel("Time")
    ax.legend()
    sns.despine()

    # plot the covariance posterior prediction
    ax = axs[1]
    if plot_discharge:
        data_lkf["q_variance_plusR"].plot(ax=ax, color=sns.color_palette()[
                                    0], label="Linear $\sigma^2_q + R$")
        data_ukf["q_variance_plusR"].plot(ax=ax, color=sns.color_palette()[
                                    1], label="Unscented $\sigma^2_q + R$")
        ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")
    else:
        ax.plot(
            np.arange(len(s_lkf.P[:, 0, 0])),
            s_lkf.P[:, 0, 0],
            c=sns.color_palette()[0],
            label="Linear $\sigma^2_S$",
        )
        ax.plot(
            np.arange(len(s_ukf.P[:, 0, 0])),
            s_ukf.P[:, 0, 0],
            c=sns.color_palette()[1],
            label="Unscented $\sigma^2_S$",
        )
        ax.set_ylabel("Storage ($S$)")
    ax.set_title("Covariance of predictions")
    ax.set_xlabel("Time")
    ax.legend()
    sns.despine()

    plt.tight_layout()
    plt.show()
    fig.savefig("/Users/tommylees/Downloads/00_simdata_cov_mean.png")


    ## Check the Uncertainty on rainfall ... (P[1,1])
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(
        np.arange(len(s_lkf.P[:, 0, 0])),
        s_lkf.P[:, 1, 1],
        c=sns.color_palette()[0],
        label="Linear $\sigma^2_r$",
    )
    ax.plot(
        np.arange(len(s_lkf.P[:, 0, 0])),
        s_ukf.P[:, 1, 1],
        c=sns.color_palette()[1],
        label="Unscented $\sigma^2_r$",
    )
    ax.legend()
    sns.despine()
    fig.savefig("/Users/tommylees/Downloads/00_rainfall_variance_together.png")


    fig, axs = plt.subplots(2, 1, figsize=(12, 4*2))
    ax = axs[0]
    ax.plot(
        np.arange(len(s_lkf.P[:, 0, 0])),
        s_lkf.P[:, 1, 1],
        c=sns.color_palette()[0],
        label="Linear $\sigma^2_r$",
    )
    ax.legend()
    sns.despine()

    ax = axs[1]
    ax.plot(
        np.arange(len(s_lkf.P[:, 0, 0])),
        s_ukf.P[:, 1, 1],
        c=sns.color_palette()[1],
        label="Unscented $\sigma^2_r$",
    )
    sns.despine()
    fig.savefig("/Users/tommylees/Downloads/00_rainfall_variance_separate.png")

    ## --- Plot Sigma Points ---

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(x=np.arange(0, 365), y=data_ukf["q_x_prior"], color=viridis)
    sigmas_h = s_ukf.sigmas_h[:, :, 0]
    for i in range(sigmas_h.shape[-1]):
        ax.scatter(x=np.arange(0, 365), y=sigmas_h[:, i], color=viridis, marker="x")

    ax.set_title("Sigma Points in Measurement Space ($\mathcal{z}_i = hx(\mathcal{y}_i)$)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge (q)")
    sns.despine()

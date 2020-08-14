"""Script containing functions for plots with the ABC model

# TRUE = black
# PRIOR = orange
# FILTERED = blue
"""
import random
from typing import Any, Optional, Tuple, List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from filterpy.common import Saver
from bookplots import (
    plot_measurements,
    plot_filter,
    plot_predictions,
    plot_track,
    plot_residual_limits,
)


def plot_possible_draws(true_data, obs_noise: float, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()
    min_bound = (true_data - (2 * np.sqrt(obs_noise))).clip(0, None)
    max_bound = true_data + (2 * np.sqrt(obs_noise))
    ax.plot(true_data, label="True")
    ax.fill_between(
        np.arange(len(true_data)), min_bound, max_bound, alpha=0.2, label="$\pm2\sigma$"
    )
    ax.set_title("$2\sigma$ for measurement noise ($\epsilon$)")
    sns.despine()

    return fig, ax


def plot_simulated_data(true_q, noisy_q, station_id: int = 39034, ax=None) -> Any:
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()
    ax.scatter(x=np.arange(len(true_q)), y=noisy_q, s=3, label="$q_{obs}$")
    ax.plot(true_q, lw=1, label="$q_{true}$")
    ax.set_xlabel("Time")
    ax.set_ylabel("Specific Discharge $mm day^{-1} km^{-2}$")
    plt.legend()
    ax.set_title(f"Simulated Data for Station: {station_id}")
    sns.despine()

    return fig, ax


def plot_filter_results(
    out: pd.DataFrame, xlim: Optional[Tuple[float]] = None, ax=None
) -> None:
    assert all(np.isin(["q_obs", "filtered", "predicted"], out.columns))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    # Labbe et al. plotting functionality
    plot_measurements(out["q_obs"], lw=0.5)
    plot_filter(out["filtered"])
    plot_predictions(out["predicted"])

    plt.legend()
    ax.set_title("Kalman Filter on Simulated Data")
    ax.set_ylabel("Specific Discharge")
    sns.despine()
    ax.set_xlim(xlim)
    return fig, ax


def plot_unobserved(out: pd.DataFrame, include_obs: bool = False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()
    out["unobserved"].plot(
        ax=ax, lw=2, ls="--", color="k", alpha=0.8, label="unobserved"
    )
    out[["filtered", "predicted"]].plot(ax=ax, lw=0.8)
    if include_obs:
        plt.scatter(
            x=np.arange(len(out["q_obs"])),
            y=out["q_obs"],
            facecolor="none",
            edgecolor="k",
            lw=0.5,
            label="Measured",
        )
    plt.legend()
    ax.set_title("The filtered values compared with the unobserved 'truth'")
    sns.despine()
    return fig, ax


def plot_obs_state_rainfall(s: Saver, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    # plot obs and state
    ax.scatter(
        np.arange(s.x.shape[0]),
        s.z[:, 1],
        label="Measured",
        color=sns.color_palette()[0],
        alpha=0.7,
    )
    ax.scatter(
        np.arange(s.x.shape[0]),
        s.x[:, 1],
        label="State Estimate",
        facecolor="None",
        edgecolor="k",
    )
    ax.set_title("$r_t$ in the State Estimate and Measured Rainfall")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rainfall ($mm day^{-1}$)")
    sns.despine()
    plt.legend()
    return fig, ax


def plot_state_storage(s: Saver, data: pd.DataFrame, ax=None, plusR: bool = False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    # plot obs and state
    ax.scatter(
        np.arange(s.x.shape[0]),
        s.x[:, 0],
        label="$S_{filtered}$",
        color=sns.color_palette()[0],
        alpha=0.6,
    )

    variance = data["s_variance_plusR"] if plusR else data["s_variance"]
    max_bound = data["s_filtered"] + (2 * np.sqrt(variance))
    min_bound = data["s_filtered"] - (2 * np.sqrt(variance))
    ax.fill_between(
        data.index,
        min_bound,
        max_bound,
        color=sns.color_palette()[0],
        alpha=0.4,
        label="$\sigma_{S}$ + R" if plusR else "$\sigma_{S}$",
    )
    ax.scatter(
        np.arange(s.x.shape[0]),
        data["S_prior"],
        label="$S_{prior}$",
        color=sns.color_palette()[1],
        alpha=0.6,
    )
    if "s_true" in [c.lower() for c in data.columns]:
        ax.scatter(
            np.arange(s.x.shape[0]),
            data["S_true"],
            label="$S_{true}$",
            facecolor="None",
            edgecolor="k",
        )

    ax.set_title("$S_t$ in the State Estimate ($x$)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Storage ($S$)")
    sns.despine()
    ax.legend()

    return fig, ax


def plot_simulated_discharge(data: pd.DataFrame, ax=None):
    assert all(np.isin(["r_obs", "q_true"], data.columns))

    if ax is None:
        fig, ax = plt.subplots()
        show_plt = False
    else:
        fig = plt.gcf()
        show_plt = True
    x = data["q_true"]

    # use prior simulations using the ABC model
    y = data["q_prior"]

    ax.scatter(x, y, marker="x", color=sns.color_palette()[1], label="Prior Discharge")

    # 1:1 line
    one_to_one_line = np.linspace(x.min(), x.max(), 10)
    ax.plot(one_to_one_line, one_to_one_line, "k--", label="1:1 Line")

    # beautifying the plot
    ax.set_xlabel("Unobserved (true) Discharge ($q_{true}$)")
    ax.set_ylabel("Prior Predicted Discharge ($q_{prior}$)")
    ax.set_title("Prior Predicted Discharge vs. True Discharge")
    ax.legend()
    sns.despine()

    return fig, ax


def plot_predicted_observed_discharge(
    data: pd.DataFrame, s: Saver, show_plt: bool = False, ax=None
):
    """Plot the unobserved discharge vs. the filtered discharge

    Args:
        data (pd.DataFrame): [description]
        s (Saver): [description]
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # x = s.z[:, 0]               # Measured Discharge
    x = data["q_true"]  # True (unobserved) Discharge
    y = (s.H @ s.x)[:, 0]  # Filtered Discharge
    ax.scatter(x, y, marker="x", label="Filtered Discharge")

    # 1:1 line
    one_to_one_line = np.linspace(x.min(), x.max(), 10)
    ax.plot(one_to_one_line, one_to_one_line, "k--", label="1:1 Line")

    # beautifying the plot
    ax.set_xlabel("Unobserved (true) Discharge ($q_{true}$)")
    ax.set_ylabel("Filtered Discharge ($Hx$)")
    ax.set_title("Filtered Discharge vs. True Discharge")
    plt.legend()
    sns.despine()

    return fig, ax


def setup_discharge_plot(ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    sns.despine()
    ax.set_title("Comparison of Predicted Discharges ($q$)")
    ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")
    ax.set_xlabel("Time (days in 2010)")
    ax.set_ylim([-0.1, 4.5])

    return fig, ax


def plot_discharge_predictions(
    data: pd.DataFrame, ax=None, filtered_prior: bool = False, plusR: bool = False
):
    assert all(np.isin(["q_prior", "q_filtered"], data.columns))
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    #
    ax.plot(
        data.index,
        data["q_prior2"] if filtered_prior else data["q_prior"],
        label="$q_{prior}$",
        ls=":",
        color=sns.color_palette()[1],
    )

    # plot the MEAN
    ax.plot(
        data.index,
        data["q_filtered"],
        label="$q_{filtered}$",
        color=sns.color_palette()[0],
    )

    # uncertainty estimate
    variance = data["q_variance_plusR"] if plusR else data["q_variance"]
    min_bound = data["q_filtered"] + (2 * np.sqrt(variance))
    max_bound = data["q_filtered"] - (2 * np.sqrt(variance))

    ax.fill_between(
        data.index,
        min_bound,
        max_bound,
        alpha=0.2,
        label="$\pm2\sigma_q$ + R" if plusR else "$\pm2\sigma_q$",
    )

    if "q_true" in data.columns:
        ax.plot(data.index, data["q_true"], label="$q_{true}$", ls="--", color="k")

    ax.scatter(
        data.index, data["q_obs"], label="$q_{obs}$", color="k", alpha=0.6, marker="x"
    )

    plt.legend()
    sns.despine()

    ax.set_title("Comparison of Predicted Discharges ($q$)")
    ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")
    ax.set_xlabel("Time")

    return fig, ax


def plot_qprior_predictions(data: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()
    ax.plot(
        data.index, data["q_prior"], color=sns.color_palette()[1], label="$q_{prior}$"
    )
    ax.set_title("Prior Predictions (without Filter)")
    ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")
    ax.set_xlabel("Time")
    sns.despine()
    plt.legend()

    return fig, ax


def plot_discharge_uncertainty(data: pd.DataFrame, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = plt.gcf()

    min_bound = data["q_filtered"] + (2 * np.sqrt(data["q_variance"]))
    max_bound = data["q_filtered"] - (2 * np.sqrt(data["q_variance"]))
    ax.plot(data["q_filtered"], label="$q_{filtered}$")
    ax.fill_between(data.index, min_bound, max_bound, alpha=0.2, label="$\pm2\sigma_q$")

    if "q_true" in data.columns:
        ax.scatter(
            data.index,
            data["q_true"],
            label="$q_{true}$",
            facecolor="None",
            edgecolor="k",
            alpha=0.6,
            s=5,
        )
    elif "q_obs" in data.columns:
        ax.scatter(
            data.index,
            data["q_obs"],
            label="$q_{obs}$",
            marker="x",
            color="k",
            alpha=0.6,
            s=5,
        )

    # beautify the plots
    sns.despine()
    plt.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")
    ax.set_title(
        "The Filtered Discharge ($Hx$) and 2 Standard Deviations ($2\sqrt{HP}$)"
    )
    if ax.get_ylim()[0] < 0:
        if ax.get_ylim()[1] > 2.85:
            ax.set_ylim((0, 4.1))
        else:
            ax.set_ylim((0, ax.get_ylim()[1]))

    return fig, ax


def plot_prior_posterior_scatter_r2(data: pd.DataFrame, s: Saver, r2: pd.DataFrame):
    fig, axs = plt.subplots(1, 3, figsize=(6 * 3, 4))
    plot_predicted_observed_discharge(data, s, ax=axs[0])
    axs[0].set_title("Filtered Discharge")
    plot_simulated_discharge(data, ax=axs[1])
    axs[1].set_title("Prior Discharge")
    sns.barplot(x="run", y="r2", data=r2, ax=axs[2])
    axs[2].set_title("$R^2$ Scores")

    if axs[2].get_ylim()[0] < -0.1:
        axs[2].set_ylim(axs[2].get_ylim()[0] - (axs[2].get_ylim()[0] * 0.05), 1)
    else:
        axs[2].set_ylim(0, 1)

    sns.despine()

    return fig, axs


def plot_filtered_true_obs(data, ax=None):
    if ax is None:
        try:
            ax = plt.gca()
        except:
            fig, ax = plt.subplots(figsize=(12, 4))

    # plot discharge (and uncertainties)
    max_ = data["q_filtered"] + np.sqrt(data["q_variance"])
    min_ = data["q_filtered"] - np.sqrt(data["q_variance"])
    ax.fill_between(data.index, min_, max_, alpha=0.3, label="$\sigma_{q}$")
    ax.plot(data["q_filtered"], ls="-", label="$q_{filtered}$")

    # plot the observations
    ax.scatter(
        data.index,
        data["q_obs"],
        ls="-",
        label="$q_{obs}$",
        facecolor="None",
        edgecolor=sns.color_palette()[1],
    )

    if "q_true" in data.columns:
        # plot the true values
        ax.plot(data["q_true"], ls="--", label="$q_{true}$", color="k")

    # beautify the plots
    plt.legend()
    sns.despine()

    return ax


def plot_raw_discharge(data: pd.DataFrame):
    assert "discharge_spec" in data.columns
    fig, ax = setup_discharge_plot()
    ax.plot(data.index, data["discharge_spec"], label="$q_{obs}$", ls="-", color="k")
    ax.set_title("CAMELS GB - Specific Discharge (2010)")
    return fig, ax


def plot_uncertainties(
    means, variances, stds: float = 3, plot_upperlower: bool = False
):
    fig, ax = plt.subplots()

    # mean estimates
    ax.plot(means, label="$\mu_t$")

    # fill between stds
    std = np.sqrt(variances) * stds
    max_ = means + std
    min_ = means - std

    if plot_upperlower:
        ax.plot(min_, color="k", ls=":", lw=1)
        ax.plot(max_, color="k", ls=":", lw=1)
    ax.fill_between(
        range(len(std)),
        min_,
        max_,
        facecolor="#ffff00",
        alpha=0.3,
        label=f"${stds}\sigma$",
    )

    ax.set_title("Estimate and Uncertainty")
    ax.legend()
    sns.despine()

    return fig, ax


def plot_std_bands(
    means, variances, stds: float = 3, plot_upperlower: bool = False, ax=None
):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = plt.gcf()

    # fill between stds
    std = np.sqrt(variances) * stds
    max_ = means + std
    min_ = means - std

    if plot_upperlower:
        ax.plot(min_, color="k", ls=":", lw=1)
        ax.plot(max_, color="k", ls=":", lw=1)
    ax.fill_between(
        range(len(std)),
        min_,
        max_,
        facecolor="#ffff00",
        alpha=0.3,
        label=f"${stds}\sigma$",
    )

    ax.set_title("Estimate and Uncertainty")
    ax.legend()
    sns.despine()

    return fig, ax

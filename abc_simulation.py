from dataclasses import dataclass
import random
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List

import numpy as np
import pandas as pd


@dataclass
class ABCSimulation:
    """Class for generating simulations from CAMELS data.

    _raw  : the underlying data from the CAMELSGB dataset
    _true : the unobserved 'true' data which we are trying to replicate
    _obs  : the noisy observations (with randomly generated noise)

    std_ : the variance of the noise (normally distributed)
    """

    data_dir: Path = Path("data")

    # known parameters for simulating uncertainty
    std_q_obs: float = 0.1
    std_r_obs: float = 0.1
    std_abc: float = 0
    std_S0: float = 0

    # set seeds
    seed: float = 1

    # optimized parameters for Evenlode (39034)
    a_true: float = 0.398887110522937
    b_true: float = 0.595108762279152
    c_true: float = 0.059819062467189064
    S0_true: float = 5.74

    def __post_init__(self):
        # set seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        # read in the initial data
        self.data = self.load_data()
        self.n_instances = len(self.data)

        # generate random 'noise'
        self.epsilon_r_obs: np.ndarray = np.random.normal(
            0, self.std_r_obs, self.n_instances
        )
        self.epsilon_q_obs: np.ndarray = np.random.normal(
            0, self.std_q_obs, self.n_instances
        )
        self.epsilon_abc: np.ndarray = np.random.normal(0, self.std_abc, 3)
        self.epsilon_S0: float = np.random.normal(0, self.std_S0)

        # generate noisy observations
        self.add_noise_to_model_parameters()

        # simulate data
        # 1. simulate true data + observation noise
        self.simulate_true_data()
        self.add_noise_to_data()

        # 2. simulate the uncertain data
        self.simulate_uncertain_data()

    def load_data(self):
        df = pd.read_csv(self.data_dir / "39034_2010.csv")
        assert all(np.isin(["precipitation", "discharge_spec"], df.columns))

        df = df.rename({"precipitation": "r_raw", "discharge_spec": "q_raw"}, axis=1)
        df["r_true"] = df["r_raw"]
        return df

    def simulate_true_data(self):
        kwargs = dict(
            a=self.a_true,
            b=self.b_true,
            c=self.c_true,
            S0=self.S0_true,
            rain=self.data["r_true"],
        )
        self.data["q_true"], self.data["S_true"] = self.abc_model(**kwargs)

    def simulate_uncertain_data(self):
        """Using the uncertain estimates of (a, b, c, S0, r)
        """
        kwargs = dict(
            a=self.a_est,
            b=self.b_est,
            c=self.c_est,
            S0=self.S0_est,
            rain=self.data["r_obs"],
        )
        self.data["q_prior"], self.data["S_prior"] = self.abc_model(**kwargs)

    def add_noise_to_model_parameters(self) -> None:
        # add normally distributed noise
        uncertain_params = (
            np.array([self.a_true, self.b_true, self.c_true]) + self.epsilon_abc
        )
        # min: 0, max: 1
        uncertain_params = np.clip(uncertain_params, a_min=0, a_max=1)

        # abc model parameters
        self.a_est = uncertain_params[0]
        self.b_est = uncertain_params[1]
        self.c_est = uncertain_params[2]

        # initial storage state (S0)
        self.S0_est = self.S0_true + self.epsilon_S0

    def add_noise_to_data(self):
        self.data["r_obs"] = np.clip(
            self.data["r_true"] + self.epsilon_r_obs, a_min=0, a_max=None
        )
        self.data["q_obs"] = np.clip(
            self.data["q_true"] + self.epsilon_q_obs, a_min=0, a_max=None
        )

    def abc_model(self, a: float, b: float, c: float, S0: float, rain: np.ndarray):
        S = S0
        q_sim = []
        S_sim = []
        for r in rain:
            q = (1 - a - b) * r + (c * S)
            S = (1 - c) * S + (a * r)
            q_sim.append(q)
            S_sim.append(S)

        return q_sim, S_sim

    def print_latex(self, double_space: bool = False):
        """
        """
        if double_space:
            space = "\\\ \\\\"  #  evaluates to -> "\\ \\"
        else:
            space = "\\\\"  #   evaluates to -> "\\"

        print("\sigma_{r\_obs}" + f"= {self.std_r_obs}")
        print(space)
        print("\sigma_{q\_obs}" + f"= {self.std_q_obs}")
        print(space)
        print("\sigma_{abc}" + f"= {self.std_abc}")
        print(space)
        print("\sigma_{S0}" + f"= {self.std_S0}")

        pass


def plot_experiment_simulation(data):
    fig, axs = plt.subplots(2, 1, figsize=(12, 4 * 2))
    ax = axs[1]
    ax.scatter(
        x=np.arange(len(data["q_obs"])),
        y=data["q_obs"],
        marker="x",
        color="k",
        lw=0.5,
        label="Measured $q_{obs}$",
    )
    data["q_true"].plot(
        ax=ax, lw=2, ls="--", color="k", alpha=0.8, label="Unobserved $q_{true}$"
    )
    data["q_prior"].plot(
        ax=ax,
        lw=2,
        ls=":",
        color=sns.color_palette()[1],
        alpha=0.8,
        label="Unfiltered Prior $q_{prior}$",
    )
    ax.set_ylabel("Specific Discharge (q - $mm day^{-1} km^{-2}$)")
    ax.set_xlabel("Time")
    ax.set_title(
        "Specific Discharge $\sigma_{q_{obs}}$: "
        f"{std_q_obs}"
        "\n$\sigma_{abc}$:"
        f"{std_abc} "
        "$\sigma_{S0}$:"
        f"{std_S0}"
        " All $\epsilon \sim N(0, \sigma)$"
    )
    ax.set_ylim(-0.1, 4.5)
    ax.legend()
    sns.despine()

    ax = axs[0]
    ax.bar(
        x=np.arange(len(data["q_obs"])),
        height=data["r_true"],
        width=1,
        label="Unobserved $r_{true}$",
    )
    ax.scatter(
        x=np.arange(len(data["q_obs"])),
        y=data["r_obs"],
        marker="x",
        color=sns.color_palette()[0],
        lw=0.5,
        label="Measured $r_{obs}$",
    )
    ax.set_ylabel("Precipitation (r - $mm day^{-1}$)")
    ax.set_xlabel("Time")
    ax.set_title("Precipitation: $\sigma_{r_{obs}}$:" f"{std_r_obs}")
    ax.legend()
    sns.despine()

    plt.tight_layout()

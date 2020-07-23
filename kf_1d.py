import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any, List
from pathlib import Path
from collections import Iterable
import random
import seaborn as sns
import matplotlib.pyplot as plt


import filterpy.kalman as kf
from bookplots import plot_measurements, plot_filter, plot_predictions


# --------------- ABC MODEL FUNCTIONS -----------------
def abcmodel(S: float, P: float) -> Tuple[float, float]:
    # hardcoding the paramters for 39034 (fit previously)
    parameters = {'a': 0.398887110522937, 'b': 0.595108762279152, 'c': 0.059819062467189064}
    a = parameters['a']
    b = parameters['b']
    c = parameters['c']

    # losses
    L = b * P

    # discharge component
    Qf = (1 - a - b) * P
    Qs = c * S
    Q_sim = Qf + Qs

    # storage component
    S = S + (a * P) - (c * S)

    return Q_sim, S


def simulate_data(
    precip: Union[pd.Series, np.ndarray, List[float]],
    measurement_noise: float = 1,
    S0: float = 5.74,
) -> pd.DataFrame:
    """Simulate data from the ABC model adding gaussian noise
    $ qsim_noise ~ N(0, sqrt(measurement_noise)) $ to the discharge
    values.

    Args:
        precip (Union[pd.Series, np.ndarray, List[float]]): Precipitation values
         to force the model.
        measurement_noise (float, optional): The variance of the gaussian noise. Defaults to 1.
        S0 (float, optional): The initial storage value. Defaults to 5.74.

    Returns:
        pd.DataFrame: DataFrame with columns
            ["precip", "qsim", "qsim_noise", "ssim"]
    """
    np.random.seed(5)
    random.seed(5)

    # initialise the arrays
    qsim = []
    qsim_noise = []
    ssim = []

    # Calculate the simulated data
    S = S0
    for p in precip:
        # simulate using the abc model
        Qsim, S = abcmodel(S, p)

        # add noise to values (clip to 0)
        Qsim_noise = np.clip(
            Qsim + np.random.normal(0, np.sqrt(measurement_noise)),
            a_min=0, a_max=None
        )

        # append simulated values
        qsim.append(Qsim)
        qsim_noise.append(Qsim_noise)
        ssim.append(S)

    sim_data = pd.DataFrame({
        "precip": precip,
        "qsim": qsim,
        "qsim_noise": qsim_noise,
        "ssim": ssim,
    })
    return sim_data


# --------------- OTHER FUNCTIONS -----------------

def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(data_dir / "39034_2010.csv")


def plot_simulated_data(data: pd.DataFrame, measurement_noise: float = 1, station_id: int = 39034) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.scatter(
        x=np.arange(len(data)),
        y=data['qsim_noise'].values,
        s=1,
        label=f"Noisy Qsim: Variance {measurement_noise}"
    )
    plt.plot(
        data['qsim'].values,
        lw=0.5,
        label="QSim without Noise"
    )
    plt.legend()
    ax.set_title(f"Simulated Data for Station: {station_id}")
    sns.despine()
    fig.savefig(f"/Users/tommylees/Downloads/simulated_data_{random.random()*10:.2f}.png")
    plt.show()


def run_1D_filter(data: pd.DataFrame, R: float, Q: float, P0: float, S0: float = 5.74) -> pd.DataFrame:
    """Run a simple 1D Kalman Filter on noisy simulated data.

    Initialize:
        We have an initial estimate of the state variance (`P0`).
        We use the first 'measurement' as our initial estimate (`X[0] = z[0]`)
    Predict:
        We require an initial input storage value - `S0`.
        We use the ABC model as our process model to produce a prediction of X (discharge).
        We assume the error/variance of this process to be `Q` (process error).
    Update:
        We use noisy observations (simulated from the ABCModel) as measurements `z`.
        We assume the noise on the simulated data to be `R` (measurement error).
        Our filtered state (X)

    Note:
        - The ABC model has an internal state parameter (S[t]) that we are currently not using.
        - The ABC model produces a direct estimate of X, rather than the change in X. We
         calculate the change in x (`dx`) as `dx = Qsim - X`, i.e. the current estimate
         minus the previous (filtered) estimate.
        - The improved state (X = discharge) estimated by the filter is not fed back into the
         ABC model.

    We use the `filterpy.kalman` implementation of the Kalman Filter.

    Args:
        data (pd.DataFrame): driving data, including precipitation (for driving the ABCmodel) &
            the noisy Qsim values. Must have columns: ['precip', 'qsim', 'qsim_noise', 'ssim']
        R (float): The measurement noise (because we have simulated data this is known)
        Q (float): The process noise (variance / uncertainty on the ABC Model estimate)
        P0 (float): The initial estimate for the state noise (variance on the prior / state uncertainty)
        S0 (float, optional): [description]. Defaults to 5.74.

    Returns:
        pd.DataFrame: [description]
    """
    assert all(np.isin(['precip', 'qsim', 'qsim_noise', 'ssim'], data.columns))

    measured_values = []
    predicted_values = []
    filtered_values = []

    kalman_gains = []
    log_likelihoods = []
    residuals = []

    # initialize step
    P = P0
    X = data["qsim_noise"][0]
    storage = S0

    # Kalman Filtered Qsim
    for ix, precip in enumerate(data["precip"]):
        # predict step
        Qsim, storage = abcmodel(storage, precip)   # process model (simulate Q)
        dx = Qsim - X                               # convert into a change (dx)
        X, P = kf.predict(x=X, P=P, u=dx, Q=Q)
        predicted_values.append(X)

        # update step
        z = data['qsim_noise'][ix]                  # measurement
        X, P, y, K, S, log_likelihood = kf.update(x=X, P=P, z=z, R=R, return_all=True)

        filtered_values.append(X)
        measured_values.append(z)

        kalman_gains.append(float(K))
        log_likelihoods.append(float(log_likelihood))
        residuals.append(float(y))

    out = pd.DataFrame({
        "measured": measured_values,
        "predicted": predicted_values,
        "filtered": filtered_values,
        "unobserved": data["qsim"],
        "kalman_gain": kalman_gains,
        "log_likelihood": log_likelihoods,
        "residual": residuals,
    })

    return out


def plot_filter_results(out: pd.DataFrame, xlim: Optional[Tuple[float]] = None) -> None:
    assert all(np.isin(["measured", "filtered", "predicted"], out.columns))

    fig, ax = plt.subplots(figsize=(12, 4))

    # Labbe et al. plotting functionality
    plot_measurements(out["measured"], lw=0.5)
    plot_filter(out["filtered"])
    plot_predictions(out["predicted"])

    plt.legend()
    ax.set_title("Kalman Filter on Simulated Data")
    ax.set_ylabel("Specific Discharge")
    sns.despine()
    ax.set_xlim(xlim)
    fig.savefig(f"/Users/tommylees/Downloads/filter_results_{random.random()*10:.2f}.png")

    plt.show()


if __name__ == "__main__":
    station_id = 39034
    original_data = read_data()
    precip_data = original_data['precipitation']

    # ------ SIMULATE DATA ------
    measurement_noise = 0.1
    data = simulate_data(precip_data, measurement_noise=measurement_noise)
    plot_simulated_data(data, measurement_noise=measurement_noise, station_id=station_id)

    # ------ HYPER PARAMETERS ------
    # kalman filter values
    P0 = 100.
    Q = 0.1
    S0 = 5.74
    R = measurement_noise

    # ------ RUN FILTER ------
    out = run_1D_filter(data, R=R, Q=Q, P0=P0, S0=S0)

    plot_filter_results(out, xlim=(0, 30))
    plot_filter_results(out, xlim=None)

    fig, ax = plt.subplots(figsize=(12, 4))
    original_data.reset_index()['discharge_spec']
    out["unobserved"].plot(ax=ax, lw=2, ls='--', color='k', alpha=0.8, label='unobserved')
    out[["filtered", "predicted"]].plot(ax=ax, lw=0.8)
    plt.legend()
    fig.savefig(f"/Users/tommylees/Downloads/unobserved_filtered_results_{random.random()*10:.2f}.png")

    ax.set_title("The filtered values compared with the unobserved 'truth'")
    sns.despine()
    # ax.set_xlim(0, 30)
    plt.show()


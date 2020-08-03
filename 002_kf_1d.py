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


def abcmodel_matrix(S0: float, P: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Convert the abc model to a matrix form (y = Ax + b)

    Calculation of Qsim.
    y = (A @ P) + (S0 * b_vec)

    Contribution from the initial storage declines over time.
    b_vec:
        [ c(1 - c) ** 0, c(1 - c) ** 1, c(1 - c) ** 2 ]

    Contribution from precipitation (diagonal) + the contribution from
     previous storage values (declines over time).
    A:
       [ (1 - a - b)    , 0.           , 0.         ]
       [ ac(1 - c) ** 0 , (1 - a - b)  , 0.         ]
       [ ac(1 - c) ** 1 , ac(1 - c)**0 , (1 - a - b)]

    Args:
        S0 (float): intial storage value
        P (np.ndarray):

    Returns:
        np.ndarray: QSim values returned as a column vector. Shape: (t, 1)
    """
    P = P[:, np.newaxis] if len(P.shape) == 1 else P
    t = len(P)  # number of timesteps

    # Calculate the contribution of initial storage.
    #   Shape: (t, 1)
    b_vec = np.array([[c * (1 - c) ** i for i in range(t)]]).T

    # Calculate the contribution of Precip / previous storage.
    #   Shape: (t, t)
    A = (np.eye(t) * (1 - a - b))

    # set the off diagonal elements:
    #   ac(1 - c) ** (row_ix + 1) - (col_x + 2)
    row_ix = np.tril_indices_from(A, k=-1)[0]
    col_ix = np.tril_indices_from(A, k=-1)[1]
    A[(row_ix, col_ix)] = a * c * (1 - c)** ((row_ix + 1) - (col_ix + 2))

    # Calculate Simulated Discharge (Q)
    Qsim = (A @ P) + (S0 * b_vec)

    assert Qsim.shape == (t, 1)
    return Qsim



def simulate_data(
    precip: Union[pd.Series, np.ndarray, List[float]],
    q_measurement_noise: float = 1,
    precip_measurement_noise: float = 1,
    S0: float = 5.74,
) -> pd.DataFrame:
    """Simulate data from the ABC model adding gaussian noise
    $ precip_noise ~ N(0, sqrt(measurement_noise)) $ to the input
    precipitation

    Args:
        precip (Union[pd.Series, np.ndarray, List[float]]): Precipitation values
         to force the model.
        measurement_noise (float, optional): The variance of the gaussian noise. Defaults to 1.
        S0 (float, optional): The initial storage value. Defaults to 5.74.

    Returns:
        pd.DataFrame: DataFrame with columns -
            ['precip', 'precip_noise', 'qsim', 'qsim_noise',
             'measured_qsim', 'ssim', 'ssim_noise']

    """
    np.random.seed(5)

    # add white noise to precip inputs
    precip_noise = np.clip(
        precip + np.random.normal(0, np.sqrt(precip_measurement_noise), len(precip)),
        a_min=0, a_max=None
    )

    # initialise the arrays
    qsim = []
    qsim_noise = []
    ssim = []
    ssim_noise = []

    # Calculate the (REAL) simulated data
    S = S0
    for p in precip:
        # simulate using the abc model
        Qsim, S = abcmodel(S, p)

        # append simulated values
        qsim.append(Qsim)
        ssim.append(S)

    # add measurement noise to the Qsim too (clip to 0)
    measured_qsim = np.clip(
        qsim + np.random.normal(0, np.sqrt(q_measurement_noise), len(qsim)),
        a_min=0, a_max=None
    )

    # Calculate the (NOISY) simulated data
    S = S0
    for p in precip_noise:
        # simulate using the abc model
        Qsim, S = abcmodel(S, p)
        qsim_noise.append(Qsim)
        ssim_noise.append(S)

    sim_data = pd.DataFrame({
        "precip": precip,
        "precip_noise": precip_noise,
        "qsim": qsim,
        "qsim_noise": qsim_noise,
        "measured_qsim": measured_qsim,
        "ssim": ssim,
        "ssim_noise": ssim_noise,
    })
    return sim_data


# --------------- OTHER FUNCTIONS -----------------

def test_matrix_vs_ordinary(P: np.ndarray, S0: float = 5.74) -> pd.DataFrame:
    # matrix
    parameters = {'a': 0.398887110522937, 'b': 0.595108762279152, 'c': 0.059819062467189064}
    qsim_matrix = abcmodel_matrix(S0=S0, P=P, **parameters)

    S = S0
    qsim_ordinary = []
    for precip in P:
        Q_sim, S = abcmodel(S, precip)
        qsim_ordinary.append(Q_sim)

    qsim_ordinary = np.array(qsim_ordinary)

    df = pd.DataFrame(
        {
            "matrix":qsim_matrix.flatten(),
            "original": qsim_ordinary.flatten()
        },
        index=np.arange(len(P))
    )

    return df


def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(data_dir / "39034_2010.csv")


def plot_simulated_data(
    data: pd.DataFrame,
    q_measurement_noise: float = 1,
    station_id: int = 39034
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    plt.scatter(
        x=np.arange(len(data)),
        y=data['measured_qsim'].values,
        s=1,
        label=f"Measured TRUE Qsim: Variance {q_measurement_noise}"
    )
    plt.plot(
        data['qsim'].values,
        lw=0.5,
        label="True QSim without Noise"
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
    assert all(np.isin(['precip', 'precip_noise', 'qsim', 'qsim_noise', 'ssim'], data.columns))

    measured_values = []
    predicted_values = []
    filtered_values = []

    kalman_gains = []
    log_likelihoods = []
    residuals = []

    # initialize step
    P = P0
    X = data["measured_qsim"][0]
    storage = S0

    # Kalman Filtered Qsim
    for ix, precip in enumerate(data["precip_noise"]):
        # predict step
        Qsim, storage = abcmodel(storage, precip)   # process model (simulate Q)
        dx = Qsim - X                               # convert into a change (dx)
        X, P = kf.predict(x=X, P=P, u=dx, Q=Q)
        predicted_values.append(X)

        # update step
        z = data['measured_qsim'][ix]                  # measurement
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


def plot_unobserved(out: pd.DataFrame, include_measured: bool = False):
    fig, ax = plt.subplots(figsize=(12, 4))
    original_data.reset_index()['discharge_spec']
    out["unobserved"].plot(ax=ax, lw=2, ls='--', color='k', alpha=0.8, label='unobserved')
    out[["filtered", "predicted"]].plot(ax=ax, lw=0.8)
    if include_measured:
        plt.scatter(x=np.arange(len(out["measured"])), y=out["measured"], facecolor='none', edgecolor='k', lw=0.5, label='Measured')
    plt.legend()
    ax.set_title("The filtered values compared with the unobserved 'truth'")
    sns.despine()
    # ax.set_xlim(0, 30)
    plt.show()

    fig.savefig(f"/Users/tommylees/Downloads/unobserved_filtered_results_{random.random()*10:.2f}.png")


if __name__ == "__main__":
    station_id = 39034
    original_data = read_data()
    precip_data = original_data['precipitation']

    # ------ SIMULATE DATA ------
    precip_measurement_noise = 3.0
    q_measurement_noise = 0.01
    data = simulate_data(
        precip_data,
        precip_measurement_noise=precip_measurement_noise,
        q_measurement_noise=q_measurement_noise
    )
    plot_simulated_data(data, q_measurement_noise=q_measurement_noise, station_id=station_id)

    # ------ HYPER PARAMETERS ------
    # kalman filter values
    P0 = initial_process_covariance = 100.
    Q = process_uncertainty = 0.001
    S0 = initial_state = 5.74
    R = measurement_uncertainty = 0.01

    # ------ RUN FILTER ------
    out = run_1D_filter(data, R=R, Q=Q, P0=P0, S0=S0)

    plot_filter_results(out, xlim=(0, 30))
    plot_filter_results(out, xlim=None)
    plot_unobserved(out, include_measured=False)


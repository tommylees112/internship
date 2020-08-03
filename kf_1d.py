import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, Union, Any, List
from pathlib import Path
from collections import Iterable
import random
import seaborn as sns
import matplotlib.pyplot as plt


import filterpy.kalman as kf
from filterpy.kalman import KalmanFilter
from bookplots import plot_measurements, plot_filter, plot_predictions, plot_track
from filterpy.common import Saver


# --------------- ABC MODEL FUNCTIONS -----------------
def abcmodel(S: float, P: float) -> Tuple[float, float]:
    # hardcoding the paramters for 39034 (fit previously)
    parameters = {
        "a": 0.398887110522937,
        "b": 0.595108762279152,
        "c": 0.059819062467189064,
    }
    a = parameters["a"]
    b = parameters["b"]
    c = parameters["c"]

    # losses
    L = b * P

    # discharge component
    Qf = (1 - a - b) * P
    Qs = c * S
    Q_sim = Qf + Qs

    # storage component
    S = S + (a * P) - (c * S)

    return Q_sim, S


def abcmodel_matrix(
    S0: float, P: np.ndarray, a: float, b: float, c: float
) -> np.ndarray:
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
    A = np.eye(t) * (1 - a - b)

    # set the off diagonal elements:
    #   ac(1 - c) ** (row_ix + 1) - (col_x + 2)
    row_ix = np.tril_indices_from(A, k=-1)[0]
    col_ix = np.tril_indices_from(A, k=-1)[1]
    A[(row_ix, col_ix)] = a * c * (1 - c) ** ((row_ix + 1) - (col_ix + 2))

    # Calculate Simulated Discharge (Q)
    Qsim = (A @ P) + (S0 * b_vec)

    assert Qsim.shape == (t, 1)
    return Qsim


def simulate_data_abc_model(
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
        a_min=0,
        a_max=None,
    )

    # initialise the arrays
    qsim = []
    qsim_noise = []
    ssim = []
    ssim_noise = []

    # Calculate the (REAL) simulated data (TRUTH)
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
        a_min=0,
        a_max=None,
    )

    # Calculate the (NOISY) simulated data (OBSERVED)
    S = S0
    for p in precip_noise:
        # simulate using the abc model
        Qsim, S = abcmodel(S, p)
        qsim_noise.append(Qsim)
        ssim_noise.append(S)

    sim_data = pd.DataFrame(
        {
            "precip": precip,
            "precip_noise": precip_noise,
            "qsim": qsim,
            "qsim_noise": qsim_noise,
            "measured_qsim": measured_qsim,
            "ssim": ssim,
            "ssim_noise": ssim_noise,
        }
    )
    return sim_data


# --------------- OTHER FUNCTIONS -----------------


def test_matrix_vs_ordinary(P: np.ndarray, S0: float = 5.74) -> pd.DataFrame:
    # matrix
    parameters = {
        "a": 0.398887110522937,
        "b": 0.595108762279152,
        "c": 0.059819062467189064,
    }
    qsim_matrix = abcmodel_matrix(S0=S0, P=P, **parameters)

    S = S0
    qsim_ordinary = []
    for precip in P:
        Q_sim, S = abcmodel(S, precip)
        qsim_ordinary.append(Q_sim)

    qsim_ordinary = np.array(qsim_ordinary)

    df = pd.DataFrame(
        {"matrix": qsim_matrix.flatten(), "original": qsim_ordinary.flatten()},
        index=np.arange(len(P)),
    )

    return df


def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(data_dir / "39034_2010.csv")


def plot_simulated_data(
    true_q, noisy_q, station_id: int = 39034, savefig: bool = True
) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.scatter(
        x=np.arange(len(data)),
        y=noisy_q,
        s=1,
        label=f"Measured",
    )
    ax.plot(true_q, lw=0.5, label="True (unobserved)")
    plt.legend()
    ax.set_title(f"Simulated Data for Station: {station_id}")
    sns.despine()

    if savefig:
        fig.savefig(
            f"/Users/tommylees/Downloads/simulated_data_{random.random()*10:.2f}.png"
        )
        plt.show()


def run_1D_filter(
    data: pd.DataFrame, R: float, Q: float, P0: float, S0: float = 5.74
) -> pd.DataFrame:
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
    assert all(
        np.isin(["precip", "precip_noise", "qsim", "qsim_noise", "ssim"], data.columns)
    )

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
        Qsim, storage = abcmodel(storage, precip)  # process model (simulate Q)
        dx = Qsim - X  #  convert into a change (dx)
        X, P = kf.predict(x=X, P=P, u=dx, Q=Q)
        predicted_values.append(X)

        # update step
        z = data["measured_qsim"][ix]  # measurement
        X, P, y, K, S, log_likelihood = kf.update(x=X, P=P, z=z, R=R, return_all=True)

        filtered_values.append(X)
        measured_values.append(z)

        kalman_gains.append(float(K))
        log_likelihoods.append(float(log_likelihood))
        residuals.append(float(y))

    out = pd.DataFrame(
        {
            "q_measured": measured_values,
            "predicted": predicted_values,
            "filtered": filtered_values,
            "unobserved": data["qsim"],
            "kalman_gain": kalman_gains,
            "log_likelihood": log_likelihoods,
            "residual": residuals,
        }
    )

    return out


def plot_filter_results(out: pd.DataFrame, xlim: Optional[Tuple[float]] = None) -> None:
    assert all(np.isin(["q_measured", "filtered", "predicted"], out.columns))

    fig, ax = plt.subplots(figsize=(12, 4))

    # Labbe et al. plotting functionality
    plot_measurements(out["q_measured"], lw=0.5)
    plot_filter(out["filtered"])
    plot_predictions(out["predicted"])

    plt.legend()
    ax.set_title("Kalman Filter on Simulated Data")
    ax.set_ylabel("Specific Discharge")
    sns.despine()
    ax.set_xlim(xlim)
    fig.savefig(
        f"/Users/tommylees/Downloads/filter_results_{random.random()*10:.2f}.png"
    )

    plt.show()


def plot_unobserved(out: pd.DataFrame, include_measured: bool = False):
    fig, ax = plt.subplots(figsize=(12, 4))
    out["unobserved"].plot(
        ax=ax, lw=2, ls="--", color="k", alpha=0.8, label="unobserved"
    )
    out[["filtered", "predicted"]].plot(ax=ax, lw=0.8)
    if include_measured:
        plt.scatter(
            x=np.arange(len(out["q_measured"])),
            y=out["q_measured"],
            facecolor="none",
            edgecolor="k",
            lw=0.5,
            label="Measured",
        )
    plt.legend()
    ax.set_title("The filtered values compared with the unobserved 'truth'")
    sns.despine()
    # ax.set_xlim(0, 30)
    plt.show()

    fig.savefig(
        f"/Users/tommylees/Downloads/unobserved_filtered_results_{random.random()*10:.2f}.png"
    )


def init_filter(
    r0: float,
    s_uncertainty: float = 1,
    r_uncertainty: float = 100,
    s_noise: float = 0.01,
    r_noise: float = 10_000,
    S0: float = 5.74,
    R: float = 1,
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
    parameters = {
        "a": 0.398887110522937,
        "b": 0.595108762279152,
        "c": 0.059819062467189064,
    }
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


def plot_measured_state_rainfall(s: Saver):
    fig, ax = plt.subplots(figsize=(12, 4))

    # plot measured and state
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
    plt.show()

    fig.savefig(
        f"/Users/tommylees/Downloads/measured_vs_state_rainfall_{random.random()*10:.2f}.png"
    )


def plot_state_storage(s: Saver):
    fig, ax = plt.subplots(figsize=(12, 4))

    # plot measured and state
    ax.scatter(
        np.arange(s.x.shape[0]),
        s.x[:, 0],
        label="State Estimate",
        facecolor="None",
        edgecolor="k",
    )
    ax.set_title("$S_t$ in the State Estimate ($x$)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Storage ($S$)")
    sns.despine()
    ax.legend()
    plt.show()

    fig.savefig(
        f"/Users/tommylees/Downloads/measured_vs_state_rainfall_{random.random()*10:.2f}.png"
    )


def plot_simulated_discharge(data: pd.DataFrame, ax=None):
    assert all(np.isin(["r_measured", "q_true"], data.columns))

    if ax is None:
        fig, ax = plt.subplots()
        show_plt = False
    else:
        fig = plt.gcf()
        show_plt = True
    x = data["q_true"]

    # simulate using the ABC model
    parameters = {
        "a": 0.398887110522937,
        "b": 0.595108762279152,
        "c": 0.059819062467189064,
    }
    y = abcmodel_matrix(S0=S0, P=data["r_measured"], **parameters)

    ax.scatter(x, y, marker='x', color=sns.color_palette()[1], label="Prior Discharge")

    # 1:1 line
    one_to_one_line = np.linspace(x.min(), x.max(), 10)
    ax.plot(one_to_one_line, one_to_one_line, "k--", label="1:1 Line")

    # beautifying the plot
    ax.set_xlabel("Unobserved (true) Discharge")
    ax.set_ylabel("ABC Simulated Discharge")
    ax.set_title("Prior Predicted Discharge vs. True Discharge")
    ax.legend()
    sns.despine()

    if show_plt:
        plt.show()



def plot_predicted_observed_discharge(data: pd.DataFrame, s: Saver, ax=None, show_plt: bool = False):
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
    y = (s.H @ s.x)[:, 0]       # Filtered Discharge
    ax.scatter(x, y, marker='x', label="Filtered Discharge")

    # 1:1 line
    one_to_one_line = np.linspace(x.min(), x.max(), 10)
    ax.plot(one_to_one_line, one_to_one_line, "k--", label="1:1 Line")

    # beautifying the plot
    ax.set_xlabel("Unobserved (true) Discharge")
    ax.set_ylabel("Filtered Discharge ($Hx$)")
    ax.set_title("Filtered Discharge vs. True Discharge")
    plt.legend()
    sns.despine()

    if show_plt:
        plt.show()
        fig.savefig(
            f"/Users/tommylees/Downloads/measured_vs_state_rainfall_{random.random()*10:.2f}.png"
        )


def abc_simulate(precip, S0: float = 5.74):
    parameters = {
        "a": 0.398887110522937,
        "b": 0.595108762279152,
        "c": 0.059819062467189064,
    }
    qsim = abcmodel_matrix(S0=S0, P=precip, **parameters)

    return qsim


if __name__ == "__main__":
    station_id = 39034
    original_data = read_data()
    precip_data = original_data["precipitation"]

    # ------ SIMULATE DATA ------
    sim_truth = True
    precip_measurement_noise = 3.0
    q_measurement_noise = 0.01
    data = original_data.copy()

    # simulate using the ABC model
    if sim_truth:
        data["q_true"] = abc_simulate(data["precipitation"])
    else:
        data["q_true"] = data["discharge_spec"]

    noise = np.random.normal(
        0, np.sqrt(q_measurement_noise), len(data["discharge_spec"])
    )
    data["q_measured"] = np.clip(data["q_true"] + noise, a_min=0, a_max=None)

    noise = np.random.normal(
        0, np.sqrt(precip_measurement_noise), len(data["discharge_spec"])
    )
    data["r_measured"] = np.clip(data["precipitation"] + noise, a_min=0, a_max=None)

    # q_true vs. q_measured
    plot_simulated_data(
        data["q_true"], data["q_measured"]
    )

    # r_true vs. r_measured
    plot_simulated_data(data["precipitation"], data["r_measured"], savefig=False)
    ax = plt.gca()
    fig = plt.gcf()
    ax.set_ylabel("Precipitation ($r$ - $mm day^{-1}$")
    ax.set_title("$r_{measured}$ vs. $r_{true}$")
    plt.show()
    fig.savefig(f"/Users/tommylees/Downloads/r_measured_graph_{random.random()*10:.2f}.png")


    # ------ HYPER PARAMETERS ------
    #  kalman filter values
    S0 = initial_state = 5.74
    R = 0.01  #  q_measurement_noise
    s_uncertainty = 1
    r_uncertainty = 100
    s_noise = 0.1
    r_noise = 10_000

    # ------ INIT FILTER ------
    kf = init_filter(
        r0=data["r_measured"][0],
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
    # for z, u in zip(data["q_measured"], data["precipitation"]):
    for z in np.vstack([data["q_measured"], data["r_measured"]]).T:
        kf.predict()
        kf.update(z)
        s.save()

    s.to_array()

    # assert np.all(
    #     pd.Series(s.z[:, 0].flatten()) == data["q_measured"]
    # ), "Expect the measured variables to be saved in the filtering process"

    # ------ INTERPRET OUTPUT ------
    # plot filtered discharge vs. the observed discharge
    fig, axs = plt.subplots(1, 2, figsize=(6*2, 4))
    plot_predicted_observed_discharge(data, s, ax=axs[0])
    plot_simulated_discharge(data, ax=axs[1])
    plt.show()
    fig.savefig(f"/Users/tommylees/Downloads/sim_pred_discharge{random.random()*10:.2f}.png")
    # plot (prior) predicted discharge vs. observed



    # Plot rainfall estimates
    plot_measured_state_rainfall(s)

    # Plot the Storage Parameter
    # TODO: (compare with "TRUE" storage)
    plot_state_storage(s)

    # fig, ax = plt.subplots(figsize=(12, 4))
    # plt.scatter(x=np.arange(s.x[:, 0].shape[0]), y=s.x[:, 0])
    # sns.despine()
    # ax.set_title("Estimated S[t]")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(12, 4))
    # plt.plot(s.P_post[:, 0, 0])
    # sns.despine()
    # ax.set_title("Variance of S[t]")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(12, 4))
    # plt.plot(s.P_post[:, 1, 1])
    # sns.despine()
    # ax.set_title("Variance of r[t]")
    # plt.show()

    # fig, ax = plt.subplots(figsize=(12, 4))
    # plot_predictions(s.x_prior[:, 0])
    # plt.plot(s.x_prior[:, 0])
    # sns.despine()
    # ax.set_title("Matrix Predictions")
    # plt.show()
# plot_track(xs[:, 0], track, zs, cov, plot_P=False,)

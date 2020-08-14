# standard library
from pathlib import Path
import datetime
import random
from typing import Dict, Tuple, Optional, Union, Any, List

# nearly standard library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Kalman Filter
import filterpy.kalman as kf
from filterpy.kalman import KalmanFilter
from filterpy.common import Saver

# optimizerrs
from scipy.optimize._constraints import Bounds
from scipy.optimize import differential_evolution
from scipy import optimize

# My code
from abc_model import PARAMETERS as parameters
from abc_plots import plot_discharge_predictions


def init_filter(
    r0: float = 0.0,
    s_variance_P00: float = 1,
    r_variance_P11: float = 100,
    s_noise_Q00: float = 0.01,
    r_noise_Q11: float = 10_000,
    S0: float = 5.74,
    q_meas_error_R00: float = 1,
    r_meas_error_R11: float = 1,
    parameters: Dict = parameters,
) -> KalmanFilter:
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
        s_variance_P00 (float, optional): [description]. Defaults to 1.
        r_variance_P11 (float, optional): [description]. Defaults to 100.
        s_noise_Q00 (float, optional): [description]. Defaults to 0.01.
        r_noise_Q11 (float, optional): [description]. Defaults to 10_000.
        s0 (float, optional): [description]. Defaults to 5.74.

    Returns:
        KalmanFilter: abc_filter
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
    abc_filter.P[:] = np.diag([s_variance_P00, r_variance_P11])

    #  state transition (F) - the process model
    # (2, 2) = square matrix
    abc_filter.F = np.array([[1 - c, a], [0.0, 1.0]])

    # Process noise (Q)
    # (2, 2) = square matrix
    abc_filter.Q = np.diag([s_noise_Q00, r_noise_Q11])

    # ------- Update Variables -------
    # Measurement function (H) (how do we go from state -> observed?)
    # (1, 2) = row vector
    abc_filter.H = np.array([[c, (1 - a - b)], [0, 1]])

    # measurement uncertainty
    # (2, 2) = square matrix OR is it just uncertainty on discharge (q)
    abc_filter.R = np.diag([q_meas_error_R00, r_meas_error_R11])

    # Control inputs (defaults)
    abc_filter.B = None  # np.ndarray([a])
    abc_filter.dim_u = 0

    return abc_filter


def run_filter(list_args: List[float], data: pd.DataFrame) -> Tuple[KalmanFilter, Saver]:
    # store the parameters in the list_args
    s_noise_Q00, r_noise_Q11, q_meas_error_R00, r_meas_error_R11 = list_args

    kf = init_filter(
        r0=0.0,
        S0=5.74,
        s_variance_P00=1,
        r_variance_P11=100,
        s_noise_Q00=s_noise_Q00,
        r_noise_Q11=r_noise_Q11,
        q_meas_error_R00=q_meas_error_R00,
        r_meas_error_R11=r_meas_error_R11,
    )

    s = Saver(kf)

    for z in np.vstack([data["q_obs"], data["r_obs"]]).T:
        kf.predict()
        kf.update(z)
        s.save()

    s.to_array()

    return kf, s


def kf_neg_log_likelihood(x: List[float], *args):
    # return the negative log likelihood of the KF
    data = args[0]
    _, s = run_filter(x, data)
    return -np.sum(s.log_likelihood)


def update_data_columns(data: pd.DataFrame, s: Saver):
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

    data["q_prior"] = ((s.H @ s.x_prior))[:, 0]

    return data


# --- FUNCTIONS --- # 
def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "39034_2010.csv")
    df["q_obs"] = df["discharge_spec"]
    df["r_obs"] = df["precipitation"]
    return df


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


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    data = read_data()
    #          Q00,         Q11,         R00,         R11
    bounds = [(1e-9, 1e7), (1e-9, 1e7), (1e-9, 1e7), (1e-9, 1e7)]
    res = differential_evolution(kf_neg_log_likelihood, bounds, args=(data, ), maxiter=10, popsize=10)

    (
            Q00,            Q11,             R00,            R11
    ) = x = [4.67129328e+05, 4.88503837e+06, 1.00000000e-09, 1.35341464e+06]
    # Q00, Q11, R00, R11 = res.x

    kf, s = run_filter([Q00, Q11, R00, R11], data)

    data = update_data_columns(data, s)

    print_latex_matrices(s)

    fig, ax = plot_discharge_predictions(
        data, filtered_prior=False, plusR=True)
    ax.set_ylim(-0.1, 4.5)

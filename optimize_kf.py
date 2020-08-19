# standard library
from pathlib import Path
import datetime
import random
import time
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

    if isinstance(data, tuple):
        # data is the first *args
        data = data[0]

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
    lls = []

    for z in np.vstack([data["q_obs"], data["r_obs"]]).T:
        kf.predict()
        kf.update(z)
        log_likelihood = kf.log_likelihood_of(z)
        lls.append(log_likelihood)
        s.save()

    s.to_array()
    lls = np.array(lls)

    return kf, s, lls


def kf_neg_log_likelihood(x: List[float], *args):
    # return the negative log likelihood of the KF
    # data = args[0]
    try:
        kf, s, log_likelihoods = run_filter(x, *args)
    except ValueError:
        # the covariance matrix must be positive semidefinite
        # TODO: why is this happening (?)
        return np.inf
    return -(log_likelihoods.sum())


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
        f"{Q[0, 0]:.4f} & 0 \\\ "
        f"0 & {Q[1, 1]:.4f}"
        "\\end{array}\\right]"
    )
    print("\\\ \\\\") # evaluates to -> "\\ \\"
    print(
        "R=\\left[\\begin{array}{cc}"
        f"{R[0, 0]:.4f} & 0 \\\ "
        f"0 & {R[1, 1]:.4f}"
        "\\end{array}\\right]"
    )


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    # --- HYPER PARAMETERS --- #
    OPTIMIZER = "de"  # "de"    "min"
    #          Q00,         Q11,         R00,         R11
    bounds = [(1e-9, 1e7), (1000, 1e7), (0.01, 1e7), (0.01, 1e7)]

    data = read_data()

    start_time = time.time()
    iso_time = time.strftime('%H:%M:%S', time.localtime(start_time))
    print(f"Running Optimizers ... {iso_time}")
    print(f"Bounds:\n\t[Q00, Q11, R00, R11] : {bounds}")

    # --- RUN OPTIMIZATION --- #
    if OPTIMIZER == "de":
        # no initial guess required
        res = differential_evolution(kf_neg_log_likelihood, bounds, args=(data, ), maxiter=100, popsize=10)
        Q00, Q11, R00, R11 = res.x

    elif OPTIMIZER == "min":
        # intial guess required
        x0 = [1, 1, 1, 1]
        res = optimize.minimize(kf_neg_log_likelihood, x0, args=(data, ), bounds=bounds, )
        Q00, Q11, R00, R11 = res.x

    else:
        print("No Optimizer Run ...")
        Q00, Q11, R00, R11 = [5.9781, 0.6406, 0.1, 0.1]

    # --- PRINT TIME TAKEN --- #
    total_time = time.time() - start_time
    iso_time = time.strftime('%H:%M:%S', time.localtime(time.time()))
    print(f"Optimizers finished ... {iso_time}")
    time_taken = f"{total_time // 60:.0f}:{total_time % 60:02.0f} min:sec" if total_time > 60 else f"{total_time} seconds"
    print(f"---- {time_taken} ----")
    print(f"Maximum Log Likelihood: {-res.fun:.2f}")

    # --- CHECK THE OPTIMIZED FILTER --- #
    # [Q00, Q11, R00, R11] = [365.837, 3.997e-06, 1e-09, 1e-09]
    # [Q00, Q11, R00, R11] = [367.3850, 0.1627, 0.0100, 0.0100],
    # [Q00, Q11, R00, R11] = [5.9781, 0.6406, 0.1000, 0.1000]
    # [Q00, Q11, R00, R11] = [5.8364, 1.9522, 1.000, 1.000]

    # params = [
    #     [365.837, 3.997e-06, 1e-09, 1e-09],
    #     [367.3850, 0.1627, 0.0100, 0.0100],
    #     [5.9781, 0.6406, 0.1000, 0.1000],
    #     [5.8364, 1.9522, 1.000, 1.000],
    # ]

    # for param in params:
        # [Q00, Q11, R00, R11] = param
    kf, s, ll = run_filter([Q00, Q11, R00, R11], data)

    data = update_data_columns(data, s)

    print_latex_matrices(s)

    fig, ax = plot_discharge_predictions(data, filtered_prior=False, plusR=True)
    ax.set_ylim(-0.1, 4.5)
    fig.savefig(f"/Users/tommylees/Downloads/data_{int(random.random() * 100)}.png")


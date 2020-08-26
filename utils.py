"""Helper functions for running kalman filters"""
from filterpy.common import Saver
import pandas as pd
import numpy as np

from abc_model import PARAMETERS as params


def print_latex_matrices(s: Saver):
    if s.Q.shape[1:] == (2, 2):
        # 2D Kalman Filter
        Q = s.Q[0, :, :]
        R = s.R[0, :, :]
        print(
            "Q=\\left[\\begin{array}{cc}"
            f"{Q[0, 0]:.4f} & 0 \\\ "
            f"0 & {Q[1, 1]:.4f}"
            "\\end{array}\\right]"
        )
        print("\\\ \\\\")  #  evaluates to -> "\\ \\"
        print(
            "R=\\left[\\begin{array}{cc}"
            f"{R[0, 0]:.4f} & 0 \\\ "
            f"0 & {R[1, 1]:.4f}"
            "\\end{array}\\right]"
        )
    elif s.Q.shape[1:] == (1, 1):
        Q = s.Q[0, :].flatten()
        R = s.R[0, :].flatten()
        # 1D Kalman Filter
        print("Q=\\left[\\begin{array}{cc}" f"{float(Q[0]):.4f}" "\\end{array}\\right]")
        print("\\\ \\\\")  #  evaluates to -> "\\ \\"
        print("R=\\left[\\begin{array}{cc}" f"{float(R[0]):.4f}" "\\end{array}\\right]")
    else:
        print(
            f"The shape of the matrices ({s.Q.shape[1:]}) has not yet been implemented"
        )


def update_data_columns(data: pd.DataFrame, s: Saver):
    # update data with POSTERIOR estimates
    # Calculate the DISCHARGE (measurement operator * \bar{x})

    if "H" in s.keys:
        # kalman filter
        data["q_filtered"] = ((s.H @ s.x))[:, 0]
        data["q_variance"] = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)))[:, 0, 0]
        data["q_variance_plusR"] = ((s.H @ s.P) @ np.transpose(s.H, (0, 2, 1)) + s.R)[
            :, 0, 0
        ]
        data["q_x_prior"] = ((s.H @ s.x_prior))[:, 0]

    elif "hx" in s.keys:
        # unscented KF = 1D state, 1D observation
        # TODO: undo the linearisation of hx
        a, b, c = params["a"], params["b"], params["c"]

        # iterate over each timestep (TODO: speed this up?)
        q_x_prior = []
        q_filtered = []
        q_variance = []
        q_variance_plusR = []
        for t_ix, r in enumerate(data["r_obs"]):
            # call the h function
            q_x_prior.append(s.hx[t_ix](s.x_prior[t_ix, 0], r))
            q_filtered.append(s.hx[t_ix](s.x[t_ix, 0], r))
            q_variance.append(s.hx[t_ix](s.P[t_ix, 0], r))
            q_variance_plusR.append(s.hx[t_ix](s.P[t_ix, 0], r) + s.R[t_ix, 0])

        data["q_x_prior"] = np.array(q_x_prior).flatten()
        data["q_filtered"] = np.array(q_filtered).flatten()
        data["q_variance"] = np.array(q_variance).flatten()
        data["q_variance_plusR"] = np.array(q_variance_plusR).flatten()

    data["s_variance"] = s.P[:, 0, 0]
    data["s_variance_plusR"] = (s.P + s.R)[:, 0, 0]
    data["s_filtered"] = s.x[:, 0]

    return data

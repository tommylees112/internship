"""ABC Model Simulation Functions
"""
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from collections import Iterable


# fit for station 39034 (fit previously)
PARAMETERS = {"a": 0.398887110522937, "b": 0.595108762279152, "c": 0.059819062467189064}


def abcmodel(S: float, P: float) -> Tuple[float, float]:
    """Simulate ONE timestep with *all parameters
    explicitly calculated. Best for learning/pedagogy.

    Args:
        S (float): Current storage
        P (float): Current precipitation

    Returns:
        Tuple[float, float]: qsim, storage
    """
    a = PARAMETERS["a"]
    b = PARAMETERS["b"]
    c = PARAMETERS["c"]

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
        P (np.ndarray): precipitation values

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


def test_matrix_vs_ordinary(P: np.ndarray, S0: float = 5.74) -> pd.DataFrame:
    # matrix
    qsim_matrix = abcmodel_matrix(S0=S0, P=P, **PARAMETERS)

    S = S0
    qsim_ordinary = []
    storages = []
    for precip in P:
        Q_sim, S = abcmodel(S, precip)
        qsim_ordinary.append(Q_sim)
        storages.append(S)

    qsim_ordinary = np.array(qsim_ordinary)
    storages = np.array(storages)

    df = pd.DataFrame(
        {
            "matrix": qsim_matrix.flatten(),
            "original": qsim_ordinary.flatten(),
            "storage": storages.flatten(),
        },
        index=np.arange(len(P)),
    )

    return df


def abc_simulate(
    precip: Iterable, S0: float = 5.74, matrix: bool = False
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Helper function to simulate using either matrix or original formulation.

    Args:
        precip ([Iterable]): [description]
        S0 (float, optional): [description]. Defaults to 5.74.
        matrix (bool, optional): [description]. Defaults to False.

    Returns:
        np.ndarray: vector of qsim values
    """
    if matrix:
        qsim = abcmodel_matrix(S0=S0, P=precip, **PARAMETERS)

        return qsim, None
    else:
        S = S0
        qsim = []
        storages = []
        for P in precip:
            Q_sim, S = abcmodel(S, P)
            qsim.append(Q_sim)
            storages.append(S)

        qsim = np.array(qsim)
        storage = np.array(storages)

        return qsim, storage

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
import seaborn as sns


# ----- Functions ----- #
def get_basin_data(basin_id: int) -> pd.DataFrame:
    """Get a dataframe for the basin

    Args:
        basin_id (int): basin integer to extract from `ds`
        ds (xr.Dataset): data with all the dynamic data stored

    Returns:
        pd.DataFrame: Dataframe with datetime index
    """
    dynamic_df = ds.sel(station_id=basin_id).to_dataframe()
    return dynamic_df


def abc_model(
    P: float,
    S: float,
    Q_tminus1: float,
    a: float,
    b: float,
    c: float,
) -> Tuple[float, float]:
    """Modelling discharge using ABC model ()

    Args:
        P (float): Precipitation
        S (float): Storage (groundwater / soil)
        Q_tminus1 (float): Discharge in previous timestep (for calculating change of Q over time)
        a (float): proportion of precipitation into storage bounds: 0-1
        b (float): proportion of precipitation lost bounds: 0-1
        c (float): proportion of storage becoming low flow bounds: 0-1

    Returns:
        dqdt (float): discharge
        dsdt (float): storage
    """
    L = b * P
    Qf = (1 - a - b) * P
    Qs = c * S
    dsdt = (a * P) - (c * S)
    Q_sim = Qf + Qs
    dqdt =  Q_sim - Q_tminus1

    return dqdt, dsdt, Q_sim


def g_h_filter(data, x0, dx, g, h, dt):
    x_est = x0
    results = []

    for z in data:
        # predict (process)
        x_pred = x_est + (dx * dt)      # prediction for X
        dx = dx

        # update (measurement)
        residual = z - x_pred

        dx = dx + h * (residual) / dt   # estimate for the rate of change in X
        x_est = x_pred + g * residual   # estimate for the raw value of X
        results.append(x_est)

    return np.array(results)


def run_abc_model(data: pd.Series, S: float, parameters: Dict) -> pd.DataFrame:
    Q_tminus1 = data["discharge_spec"][0]
    dQ_vector = []
    dS_vector = []
    qsims = []
    ssims = []
    init_qsims = []

    for Q, P in zip(data["discharge_spec"][1:], data["precipitation"][1:]):
        # calculate the values
        dq, ds, Q_sim = abc_model(
            P=P, S=S, Q_tminus1=Q_tminus1,
            **parameters
        )

        # update estimates
        qsim = Q_tminus1 + dq
        S = S + ds
        Q_tminus1 = Q

        # append the results
        dQ_vector.append(dq)
        dS_vector.append(ds)
        qsims.append(qsim)
        ssims.append(S)
        init_qsims.append(Q_sim)

    # create output dataframe
    out = pd.DataFrame({
        "dqdt": dQ_vector,
        "dsdt": dS_vector,
        "qsim": qsims,
        "ssim": ssims,
        # "init_qsim": init_qsims,
    })
    out.index = data.index[1:]
    out["qobs"] = data["discharge_spec"][1:]

    return out


if __name__ == "__main__":
    # ----- Params ----- #
    train_start_date = "01-01-1980"
    train_end_date = "31-12-1989"
    test_start_date = "01-01-1990"
    test_end_date = "31-12-2010"
    station_id = 39034  # Evenlode at Cassington Mill

    # ----- Data ----- #
    data_dir = Path("/Volumes/Lees_Extend/data/ecmwf_sowc/data/interim")
    ds = xr.open_dataset(data_dir / "camels_preprocessed/data.nc")
    static = xr.open_dataset(data_dir / "static/data.nc")

    df = get_basin_data(basin_id=station_id).loc[:, ["precipitation", "discharge_spec"]]
    static_ds = static.sel(station_id=station_id)

    # train test split
    train = df.loc["01-01-1980": "31-12-1989"]
    test = df.loc["01-01-1990": "31-12-2010"]

    data = df.loc['2010']

    # ----- Model (fit using RRMPG) ----- #
    # best fit parameters
    parameters = {'a': 0.398887110522937, 'b': 0.595108762279152, 'c': 0.059819062467189064}
    S = 6.753516938293166  # (selected as mean of test period storage)

    # ----- Model Run (no noise in observations) ----- #
    out = run_abc_model(data=data, S=S, parameters=parameters)

    fig, ax = plt.subplots(figsize=(12, 4))
    out[["qobs", "qsim"]].plot(ax=ax)
    ax.set_title(f"{station_id}: Discharge for 2010")
    ax.set_xlabel('Time')
    ax.set_ylabel('Specific Discharge ($mm^{3} day^{-1} km^{-2}$)')
    ax.legend()
    sns.despine()

    out[["qobs", "qsim"]].plot(ax=ax)

    # add some noise to the observations (Q, P ~ N(0, 1))
    random.seed(5)
    np.random.seed(5)

    noisy_data = data.copy()
    noisy_data["discharge_true"] = noisy_data["discharge_spec"]
    noisy_data["precip_true"] = noisy_data["precipitation"]
    noisy_data["discharge_spec"] = noisy_data["discharge_spec"] + np.random.normal(0, 1)
    noisy_data["precipitation"] = noisy_data["precipitation"] + np.random.normal(0, 1)

    out = run_abc_model(data=data, S=S, parameters=parameters)


X0 = noisy_data["discharge_true"][0]
P0 = 1**2

x = gaussian(X0, P0)
for ix, z in enumerate(noisy_data["discharge_spec"][1:]):
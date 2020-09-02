import numpy as np
from kf_2d import read_data, abc_simulate


if __name__ == "__main__":
    data = read_data()
    rename = dict(precipitation="r", discharge_spec="y",)
    data = data.rename(rename, axis=1)

    # generate the target / input data
    data["y_hat"], data["S_hat"] = abc_simulate(data["r"])
    data["S_hat"] = data["S_hat"].shift(1)
    data.loc[0, "S_hat"] = 5.74  #  default S0
    data["y_t-1"] = data["y"].shift(1)
    data["target"] = data["y"] - data["y_hat"]
    data["input"] = tuple(
        zip(data["y_t-1"].values, data["r"].values, data["y_hat"].values)
    )

"""
d = data.copy()
d = d.rename(
    dict(precipitation="r", discharge_spec="y", S_prior="S", q_prior="y_hat",),
    axis=1,
)
d["y_t-1"] = d["y"].shift(1)
d = d[["time", "y", "r", "S", "y_hat", "y_t-1"]]
d["target"] = d["y"] - d["y_hat"]
d["input"] = tuple(zip(d["y_t-1"], d["y_hat"], d["r"]))
"""

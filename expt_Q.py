from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import seaborn as sns

from kf_2d import run_kf, simulate_data, read_data, create_rundir, calculate_r2_metrics
from abc_plots import (plot_prior_posterior_scatter_r2, plot_discharge_uncertainty, plot_filtered_true_obs)
from matplotlib import cm
import imageio


def convert_to_image(fig) -> np.ndarray:
    """Return a figure canvas as a numpy array (RGB)

    Args:
        fig ([type]): matplotlib figure object to convert

    Returns:
        [np.ndarray (height, width, 3)]: 0-255 RGB values as an array
    """
    # Used to return the plot as an image array
    fig.canvas.draw()       # draw the canvas, cache the renderer
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return image


# Set seeds (for reproducibility)
random.seed(1)
np.random.seed(1)


@dataclass
class Experiment:
    experiment_name: str
    run_dir: Path

    R: float
    # PROCESS
    S0: float
    P00 = s_variance: float
    P11 = r_variance: float
    Q00 = s_noise: float
    Q01: float = 0
    Q11 = r_noise: float
    Q10: float = 0

    # How often to make observations?
    observe_every: int

    def __post_init__(self):
        # get plot_dir
        self.plot_dir = self.get_plot_dir()

    def get_plot_dir(self) -> Path:
        self.plot_dir = self.run_dir / f"plots"
        if not self.plot_dir.exists():
            self.plot_dir.mkdir(parents=True, exist_ok=True)

        return self.plot_dir

    def run(self):
        print(f"Running Experiment {experiment_name}")
        # Create Simulated Data
        original_data = read_data()
        self.data = simulate_data(
            original_data=original_data,
            q_obs_noise=self.q_obs_noise,
            r_obs_noise=self.r_obs_noise
        )

        # Run Kalman Filter
        self.kf, self.s, self.data = run_kf(
            data=self.data,
            s_variance=self.s_variance,
            r_variance=self.r_variance,
            s_noise=self.s_noise,
            r_noise=self.r_noise,
            R=self.R,
            S0=self.S0,
            observe_every=self.observe_every,
        )

        r2 = calculate_r2_metrics(self.data)
        self.r2 = r2
        self.prior_r2 = float(r2.loc[r2['run'] == 'prior', 'r2'])
        self.posterior_r2 = float(r2.loc[r2['run'] == 'posterior', 'r2'])

        self.plot_main_plots()

    def plot_main_plots(self):
        fig, axs = plot_prior_posterior_scatter_r2(self.data, self.s, self.r2)
        fig.savefig(self.plot_dir / f"001_{self.observe_every}_discharge_scatter.png")

        fig, ax = plot_discharge_uncertainty(self.data)
        fig.savefig(self.plot_dir / f"002_{self.observe_every}_discharge_uncertainty.png")

        fig, ax = plt.subplots(figsize=(12, 4))
        ax = plot_filtered_true_obs(e.data)
        ax.set_title(f"Observe Every: {e.observe_every}")
        ax.set_ylim(-1, 3.1)
        fig.savefig(e.plot_dir / f"003_{e.observe_every}_filtered_true_obs.png")

        plt.close('all')


if __name__ == "__main__":
    experiment_name = "change_Q"
    run_dir = Path("runs")
    data_dir = Path("data")

    experiment_run = create_rundir(run_dir, experiment_name)

    # get dataframe of R2s
    posterior_r2s = [e.posterior_r2 for e in experiments]
    prior_r2s = [e.prior_r2 for e in experiments]
    delta_r2s = [post_r2 - prior_r2 for (post_r2, prior_r2) in zip(posterior_r2s, prior_r2s)]

    r2s = pd.DataFrame({
        "posterior": posterior_r2s,
        "prior": prior_r2s,
        "delta": delta_r2s,
    })
    r2s["observe_every"] = r2s.index + 1

    # plot the increase in variance between observations
    q_vars = [e.data["q_variance"] for e in experiments]
    viridis = cm.get_cmap('viridis', 12)
    colors = [viridis(i) for i in np.linspace(0, 1, 10)]
    fig, ax = plt.subplots(figsize=(12, 8))
    for ix, q_var in enumerate(q_vars):
        ax.plot(q_var, alpha=0.7, color=colors[ix], label=f"Observe Every: {ix+1}")

    plt.yscale('log')
    plt.legend()
    sns.despine()

    # Create gif
    figs = []
    images = []
    for e in experiments:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax = plot_filtered_true_obs(e)
        figs.append(fig)
        # kwargs_write = {'fps':1.0, 'quantizer':'nq'}
    plt.close('all')

    imageio.mimsave(e.plot_dir / '003_all.gif', [convert_to_image(fig) for fig in figs], fps=1)

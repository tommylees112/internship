import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Any, Dict, Callable, Union, Generator, Iterator


# --------- IO ---------  #
def read_data(data_dir: Path = Path("data")) -> pd.DataFrame:
    return pd.read_csv(data_dir / "39034_2010.csv")


# --------- Differential Evolution ---------  #
fobj = lambda x: sum(x ** 2) / len(x)


def scatter_plot_function(fobj, n_iters: int = 10000):
    # randomly generate inputs to fobj
    d = np.linspace(-1, 1, 1000)
    inputs = np.array([np.random.choice(d, 2) for _ in range(n_iters)])
    zs = np.array([fobj(x) for x in inputs])
    xs = inputs[:, 0]
    ys = inputs[:, 1]

    # plot scatter of x,y,z
    fig, ax = plt.subplots()
    c = ax.scatter(xs, ys, c=zs)
    cbar = plt.colorbar(c)
    cbar.set_label("Value of f")
    ax.set_title("Evaluating the Function: $\sum_{i=1}^{N}\\frac{x_i^2}{N}$")

    return fig, ax


def de(
    fobj: Callable,
    bounds: List[Tuple[float, float], ...],
    mut: float = 0.8,
    crossp: float = 0.7,
    popsize: int = 20,
    its: int = 1000,
    yield_best: bool = True,
) -> Union[Iterator[Tuple[np.ndarray, float]], Iterator[Tuple[np.ndarray, float, int]]]:
    # NOTE: type annotation Generator[yield_type, send_type, return_type] or Iterator
    #  --- INITIALISATION --- #
    dimensions = len(bounds)
    # generate uniform random numbers 0-1
    pop = np.random.rand(popsize, dimensions)
    # scale to min/max
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff

    #  --- EVALUATION --- #
    # determine the fittest of each individual
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    for i in range(its):
        # mutate, recombine, replace for each individual in the population
        for target_idx in range(popsize):
            # --- MUTATION --- #
            idxs = [idx for idx in range(popsize) if idx != target_idx]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            # ensure one value is always mutated
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True

            # --- RECOMBINATION --- #
            trial = np.where(cross_points, mutant, pop[target_idx])
            trial_denorm = min_b + trial * diff

            # --- REPLACEMENT --- #
            # is the trial better than the target ?
            f = fobj(trial_denorm)
            if f < fitness[target_idx]:
                fitness[target_idx] = f
                pop[target_idx] = trial
                if f < fitness[best_idx]:
                    best_idx = target_idx
                    best = trial_denorm
        if yield_best:
            yield best, fitness[best_idx]
        else:  #  yield the whole population
            yield min_b + pop * diff, fitness, best_idx


if __name__ == "__main__":
    df = read_data()

    # check the function calls
    bounds = [(1, 4), (1, 2)]
    mut = 0.8
    crossp = 0.7
    popsize = 20
    its = 1000

    dimensions = len(bounds)
    pop = np.random.rand(popsize, dimensions)
    min_b, max_b = np.asarray(bounds).T
    diff = np.fabs(min_b - max_b)
    pop_denorm = min_b + pop * diff
    fitness = np.asarray([fobj(ind) for ind in pop_denorm])
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]

    i = 0
    individual_idx = 0

    # for i in range(its):
    # for individual_idx in range(popsize):
    # get all of the indexes of OTHER individuals (not the pop)
    other_idxs = [idx for idx in range(popsize) if idx != individual_idx]
    # randomly select 3 OTHER individuals from the 0,1 scaled pop
    a, b, c = pop[np.random.choice(other_idxs, 3, replace=False)]

    mutant = np.clip(a + mut * (b - c), 0, 1)
    cross_points = np.random.rand(dimensions) < crossp
    if not np.any(cross_points):
        cross_points[np.random.randint(0, dimensions)] = True
    trial = np.where(cross_points, mutant, pop[individual_idx])
    trial_denorm = min_b + trial * diff
    f = fobj(trial_denorm)
    if f < fitness[individual_idx]:
        fitness[individual_idx] = f
        pop[individual_idx] = trial
        if f < fitness[best_idx]:
            best_idx = individual_idx
            best = trial_denorm
        # yield best, fitness[best_idx]

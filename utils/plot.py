import numpy as np
from typing import Optional
from matplotlib import pyplot as plt


def ignore_nan(x):
    series = np.array(x).astype(np.double)
    mask = np.isfinite(series)
    xs = np.arange(len(x))
    return xs[mask], series[mask]


def plot_perf(
    train_perf: Optional[list] = None,
    val_perf: Optional[list] = None,
    title: Optional[str] = None,
    save: bool = False,
):
    fig, subplot = plt.subplots(
        1,
        figsize=(10, 5),
        dpi=80,
        sharex=True,
    )

    if title:
        subplot.set_title(title)

    if train_perf:
        subplot.plot(
            train_perf,
            marker="o",
            label="train",
            color="steelblue",
        )

    if val_perf:
        xs, series = ignore_nan(val_perf)
        subplot.plot(xs, series, marker="o", label="val", color="orange")

    subplot.legend(loc="upper left")
    subplot.set_xlabel("Epoch")

    plt.plot()
    plt.pause(0.01)

    if save:
        return fig


def plot_perf_one_curve(
    perf: Optional[list] = None,
    title: Optional[str] = None,
    save: bool = False,
    colour="steelblue", # steelblue for training, orange for validation.
):
    fig, subplot = plt.subplots(
        1,
        figsize=(10, 5),
        dpi=80,
        sharex=True,
    )

    if title:
        subplot.set_title(title)

    subplot.plot(
        perf,
        marker="o",
        color=colour,
    )

    subplot.set_xlabel("Epoch")

    plt.plot()
    plt.pause(0.01)

    if save:
        return fig

from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis.utils import analyse_cross_ff, cross_fcn_sloped


def plot_cross_result():
    t = np.linspace(0, 50, num=101, endpoint=True)
    slope_ij_multiple = [1, 6, 3, 1, 6]
    C_ij_multiple = [5, 6, 15, 20]
    R_ij_multiple = [5, 7, 1, 2, 1]
    intercept = 0
    C = [4]
    slope = [2, 1]
    R = [5, 2]
    fig, axs = plt.subplots(1, 2, figsize=(30, 10), sharey=False)
    change_cross_value = partial(cross_fcn_sloped, crossover_amount=2)
    change_single_cross_value = partial(cross_fcn_sloped, crossover_amount=1)
    axs[0].axhline(y=0, color="r", linestyle="--", label="y0")
    axs[0].plot(
        t,
        change_cross_value(
            t,
            intercept,
            C_ij_multiple,
            slope_ij_multiple,
            R_ij_multiple,
            crossover_amount=2,
        ),
        label="multiple crossovers",
    )
    axs[0].axhline(y=0, color="b", linestyle="--", label="y0")
    axs[1].plot(
        t,
        change_single_cross_value(t, 0, C, slope, R, crossover_amount=1),
        label="single crossover",
    )

    plt.plot()
    plt.grid()
    plt.legend()
    plt.xlim(0, 45)
    plt.show()


def plot_ff(
    hs: np.ndarray,
    S: np.ndarray,
    residuals,
    y_0: np.ndarray,
    R: np.ndarray,
    crossovers: np.ndarray,
    slope_current: np.ndarray,
    crossover_amount: int,
    title="F(S)",
):
    if len(residuals.shape) == 1:
        residuals = np.expand_dims(residuals, -1)

    fig, ax = plt.subplots(figsize=(30, 10))
    ax.set_title(title)
    ax.set_title(title)
    C = np.log10(crossovers)
    fit_func = 10 ** cross_fcn_sloped(
        np.log10(S),
        y_0,
        C[0],
        C[1],
        slope_current[0],
        slope_current[1],
        slope_current[2],
        R[0],
        R[1],
        R[2],
        crossover_amount=2,
    )
    ax.errorbar(
        S,
        fit_func,
        fmt="g--",
        capsize=7,
        yerr=2 * np.std(residuals, axis=0),
        label=r"$F(S) \pm 2\sigma$",
    )

    S_new = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    colors = ["blue", "green", "red", "purple"]
    cross_log = np.log10(crossovers)
    mask = np.log10(S_new) <= cross_log[0]

    ax.plot(
        S_new[mask],
        hs[mask],
        ".",
        color=colors[0],
        label=rf"$H_0(S) \sim {slope_current[0]:.2f} \cdot S$",
    )

    if len(crossovers) > 1:
        mask1 = (np.log10(S_new) > cross_log[0]) & (np.log10(S_new) <= cross_log[1])
        ax.plot(
            S_new[mask1],
            hs[mask1],
            ".",
            color=colors[1],
            label=rf"$H_1(S) \sim {slope_current[1]:.2f} \cdot S$",
        )
    mask2 = np.log10(S_new) > cross_log[-1]

    ax.plot(
        S_new[mask2],
        hs[mask2],
        ".",
        color=colors[2],
        label=rf"$H_2(S) \sim {slope_current[2]:.2f}  \cdot S$",
    )
    for c in crossovers:
        ax.axvline(c, color="k", linestyle="--", label=f"Cross at $S={c}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()

    plt.show()

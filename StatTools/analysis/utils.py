from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class var_estimation:
    value: float
    stderr: float


@dataclass
class var_estimation_multiple:
    multiple_value: np.ndarray
    multiple_stderr: np.ndarray


@dataclass
class dfa_params:
    intercept: var_estimation
    cross: var_estimation
    slope_l: var_estimation
    slope_h: var_estimation
    ridigity: var_estimation
    max_ridigity: float = +np.inf
    min_ridigity: int = 1
    max_intercept: int = +np.inf
    min_intercept: float = -np.inf
    max_slope_l: int = 5
    min_slope_l: int = 0
    max_slope_h: int = 5
    min_slope_h: int = 0
    ridigity_initial_parameter: int = 5
    intercept_initial_parameter: int = 0
    slope_l_initial_parameter: int = 1
    slope_h_initial_parameter: int = 1


@dataclass
class multiple_dfa_params:
    multiple_intercept: var_estimation_multiple
    multiple_cross: var_estimation_multiple
    multiple_slope_current: var_estimation_multiple
    multiple_ridigity: var_estimation_multiple


def f_fcn_without_ooverflaw(x):
    return np.maximum(0, x) + np.log1p(np.exp(-np.abs(x)))


def f_fcn(x, R, C):
    """
    ReLu type function with ridigity-R, C-slope scale
    """
    dx = x - C
    log_new = f_fcn_without_ooverflaw(R * dx)
    return (log_new / R) * dx / np.sqrt(1 + dx**2)


def rev_f_fcn(x, R, C):
    """
    Invert Relu type function
    """
    dx = x - C
    log_new = f_fcn_without_ooverflaw(R * dx)
    return (log_new / R) * (-dx) / np.sqrt(1 + dx**2)


def tf_minus_inf(x, R, C1):
    return rev_f_fcn(x, R, C1)


def tf_plus_inf(x, R, C2):
    return f_fcn(x, R, C2)


def tf(x, R, C1, C2):
    """
    Step function
    """
    if C1 <= float("-inf"):
        return tf_minus_inf(x, R, C1)
    elif C2 >= float("inf"):
        return tf_plus_inf(x, R, C2)
    else:
        return -f_fcn(x, R, C2) - rev_f_fcn(x, R, C1)


def single_cross_fcn_sloped(x, y_0, C_12, slope_1, slope_2, R_12):

    value = None
    value = (slope_1 * tf(0, R_12, -100, C_12)) - (slope_2 * tf(0, R_12, C_12, 100))
    func1 = slope_1 * (tf(x, R_12, -100, C_12))
    func2 = slope_2 * tf(x, R_12, C_12, 100)
    return y_0 + func1 + func2 - value


def multiple_cross_fcn_sloped(
    x, y_0, C_ij: np.ndarray, slope_ij: np.ndarray, R_ij: np.ndarray
):
    curr_C = None
    prev_C = -100
    result_sloped = 0
    result = np.zeros_like(x, dtype=float)
    for index in range(len(slope_ij)):
        slope = slope_ij[index]
        R = R_ij[index]
        if index < len(C_ij):
            curr_C = C_ij[index]
        else:
            curr_C = 100
            pass
        result += slope * tf(x, R, prev_C, curr_C)
        result_sloped += slope * tf(0, R, prev_C, curr_C)
        prev_C = curr_C
    return y_0 + result - result_sloped


def scalar_for_fitting(x, y0, C1, C2, s0, s1, s2, R1, R2, R3):
    C_ij = np.array([C1, C2])
    slope_ij = np.array([s0, s1, s2])
    R_ij = np.array([R1, R2, R3])
    return multiple_cross_fcn_sloped(x, y0, C_ij, slope_ij, R_ij)


def single_cross_fcn(x, y_0, C_12, slope_1, slope_2, R_12):
    """
    Creates piesewise linear function with slopes definition
    """
    return y_0 + slope_1 * tf(x, R_12, -100, C_12) + slope_2 * tf(x, R_12, C_12, 100)


def analyse_ff(hs: np.ndarray, S: np.ndarray):
    """
    Analyses real data: F(s) and s and simulated data with linear regression's model.
    """
    hs = hs
    S = S
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T

    (intercept, cross, slope_l, slope_h, ridigity), pcov, infodict, mesg, ier = (
        curve_fit(
            single_cross_fcn,
            np.log10(s.flatten()),
            np.log10(hs.flatten()),
            p0=(
                dfa_params.intercept_initial_parameter,
                np.log10(S[len(S) // 2]),
                dfa_params.slope_l_initial_parameter,
                dfa_params.slope_h_initial_parameter,
                dfa_params.ridigity_initial_parameter,
            ),
            bounds=(
                [
                    dfa_params.min_intercept,
                    np.log10(S[0]),
                    dfa_params.min_slope_l,
                    dfa_params.min_slope_h,
                    dfa_params.min_ridigity,
                ],
                [
                    dfa_params.max_intercept,
                    np.log10(S[-1]),
                    dfa_params.max_slope_l,
                    dfa_params.max_slope_h,
                    dfa_params.max_ridigity,
                ],
            ),
            full_output=True,
            maxfev=1000,
            nan_policy="raise",
        )
    )

    return (
        dfa_params(
            intercept=var_estimation(value=intercept, stderr=np.sqrt(pcov[0][0])),
            cross=var_estimation(value=10**cross, stderr=np.sqrt(pcov[1][1])),
            slope_l=var_estimation(value=slope_l, stderr=np.sqrt(pcov[2][2])),
            slope_h=var_estimation(value=slope_h, stderr=np.sqrt(pcov[3][3])),
            ridigity=var_estimation(value=ridigity, stderr=np.sqrt(pcov[4][4])),
        ),
        10
        ** single_cross_fcn(np.log10(s), intercept, cross, slope_l, slope_h, ridigity)
        - hs,
    )


def analyse_multiple_cross_ff(hs: np.ndarray, S: np.ndarray):
    """
    Analyses real data: F(s) and s and simulated data with linear regression's model.
    """
    hs = hs
    S = S
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    po = (0, np.log10(S[len(S) // 3]), np.log10(S[2 * len(S) // 3]), 1, 1, 1, 5, 5, 5)
    bounds_min = (-np.inf, np.log10(S[0]), np.log10(S[1]), 0, 0, 0, 1, 1, 1)
    bounds_max = (1, np.log10(S[-2]), np.log10(S[-1]), 5, 5, 5, 20, 20, 20)
    popt, pcov, infodict, mesg, ier = curve_fit(
        scalar_for_fitting,
        np.log10(s.flatten()),
        np.log10(hs.flatten()),
        p0=po,
        bounds=(
            bounds_min,
            bounds_max,
        ),
        full_output=True,
        maxfev=6000,
        nan_policy="raise",
    )
    y0, C1, C2, s0, s1, s2, R1, R2, R3 = popt
    stderr = np.sqrt(np.diag(pcov))
    err_y0, err_C1, err_C2, err_s0, err_s1, err_s2, err_R1, err_R2, err_R3 = stderr
    intercept_values = np.array([y0])
    intercept_err = np.array([y0])
    cross_values = np.array([10**C1, 10**C2])
    cross_err = np.array([err_C1, err_C2])
    slope_values = np.array([s0, s1, s2])
    slope_errs = np.array([err_s0, err_s1, err_s2])
    ridigity_values = np.array([R1, R2, R3])
    ridigity_err = np.array([err_R1, err_R2, err_R3])

    return (
        multiple_dfa_params(
            multiple_intercept=var_estimation_multiple(
                multiple_value=intercept_values, multiple_stderr=intercept_err
            ),
            multiple_cross=var_estimation_multiple(
                multiple_value=cross_values,
                multiple_stderr=cross_err,
            ),
            multiple_slope_current=var_estimation_multiple(
                multiple_value=slope_values,
                multiple_stderr=slope_errs,
            ),
            multiple_ridigity=var_estimation_multiple(
                multiple_value=ridigity_values,
                multiple_stderr=ridigity_err,
            ),
        ),
        10 ** scalar_for_fitting(np.log10(S), y0, C1, C2, s0, s1, s2, R1, R2, R3) - hs,
    )


def analyse_plot_ff(hs, S, params: dfa_params, residuals: np.ndarray, title="F(S)"):
    """
    Analyse plot that shows comparison
    linear regression's model's data plot and real data: F(s)~s where F(s)-fluctuation functions
    """
    if len(residuals.shape) == 1:
        residuals = np.expand_dims(residuals, -1)
    cross = params.cross.value
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    ax.errorbar(
        S,
        10
        ** (
            single_cross_fcn(
                np.log10(S),
                params.intercept.value,
                np.log10(params.cross.value),
                params.slope_l.value,
                params.slope_h.value,
                params.ridigity.value,
            )
        ),
        fmt="g--",
        capsize=7,
        yerr=2 * np.std(residuals, axis=0),
        label=f"$F(S) \pm 2 \sigma$",
    )
    S = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    ax.plot(
        S[S <= cross],
        hs[S <= cross],
        ".",
        color="blue",
        label=f"$H_l(S) \sim {params.slope_l.value:.2f} \pm {2*params.slope_l.stderr:.2f} S$",
    )
    ax.plot(
        S[S > cross],
        hs[S > cross],
        ".",
        color="red",
        label=f"$H_h(S)  \sim {params.slope_h.value:.2f} \pm {2*params.slope_h.stderr:.2f} S$",
    )
    ax.axvline(
        params.cross.value,
        label=f"Cross, $S = {params.cross.value:.2f} \pm {params.cross.stderr:.2f}$",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()
    plt.plot()
    plt.show()


def analyse_multiple_plot_ff(
    hs: np.ndarray,
    S: np.ndarray,
    params: multiple_dfa_params,
    residuals: np.ndarray,
    title="F(S)",
):
    if len(residuals.shape) == 1:
        residuals = np.expand_dims(residuals, -1)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title)
    y0 = params.multiple_intercept.multiple_value
    crossovers = params.multiple_cross.multiple_value
    slope_current = params.multiple_slope_current.multiple_value
    R = params.multiple_ridigity.multiple_value
    fit_func = 10 ** multiple_cross_fcn_sloped(
        np.log10(S), y0, np.log10(crossovers), slope_current, R
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
        label=rf"$H_0(S) \sim {slope_current[0]:.2f} \pm {2*params.multiple_slope_current.multiple_stderr[0]:.2f} \cdot S$",
    )
    if len(crossovers) > 1:
        mask1 = (np.log10(S_new) > cross_log[0]) & (np.log10(S_new) <= cross_log[1])
        ax.plot(
            S_new[mask1],
            hs[mask1],
            ".",
            color=colors[1],
            label=rf"$H_1(S) \sim {slope_current[1]:.2f} \pm {2*params.multiple_slope_current.multiple_stderr[1]:.2f} \cdot S$",
        )
    mask2 = np.log10(S_new) > cross_log[-1]
    ax.plot(
        S_new[mask2],
        hs[mask2],
        ".",
        color=colors[2],
        label=rf"$H_2(S) \sim {slope_current[2]:.2f} \pm {2*params.multiple_slope_current.multiple_stderr[2]:.2f} \cdot S$",
    )
    for c in crossovers:
        ax.axvline(c, color="k", linestyle="--", label=f"Cross at $S={c}$")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.legend()
    plt.show()

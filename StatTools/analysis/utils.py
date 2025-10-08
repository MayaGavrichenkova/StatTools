from dataclasses import dataclass

import numpy as np
from scipy.optimize import curve_fit


@dataclass
class var_estimation:
    value: float
    stderr: float


@dataclass
class dfa_params_linear:
    intercept: var_estimation
    slope_l: var_estimation


@dataclass
class dfa_params:
    intercept: var_estimation
    cross: var_estimation
    slope_l: var_estimation
    slope_h: var_estimation
    ridigity: var_estimation


def f_fcn(x, R, C):
    """
    ReLu type function with ridigity-R, C-slope scale
    """
    return np.log(1 + np.exp(R * (x - C))) / R * (x - C) / np.sqrt(1 + (x - C) ** 2)


def rev_f_fcn(x, R, C):
    """
    Invert Relu type function
    """
    return np.log(1 + np.exp(R * (x - C))) / R * (-(x - C)) / np.sqrt(1 + (x - C) ** 2)


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
    prev_slope = None
    prev_R = None
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
        result += slope * tf(x, R, prev_C, curr_C)
        result_sloped += slope * tf(0, R, prev_C, curr_C)
        prev_C = curr_C
    return y_0 + result - result_sloped


def single_cross_fcn(x, y_0, C_12, slope_1, slope_2, R_12):
    """
    Creates piesewise linear function with slopes definition
    """
    return y_0 + slope_1 * tf(x, R_12, -100, C_12) + slope_2 * tf(x, R_12, C_12, 100)


def analyse_ff(hs, S):
    """
    Analyses real data: F(s) and s and simulated data with linear regression's model.
    """
    hs = hs
    S = S
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T

    x_data = np.log10(s.flatten())
    y_data = np.log10(hs.flatten())

    def fit_func(x, cross, slope_l, slope_h, rigidity):
        """
        Сreates linear regression's model
        """
        return single_cross_fcn_sloped(x, cross, slope_l, slope_h, rigidity)

    p0 = (0, np.log10(S[len(S) // 2]), 1, 1, 5)
    bounds = (
        [-np.inf, np.log10(S[0]), 0, 0, 1],
        [+np.inf, np.log10(S[-1]), 5, 5, +np.inf],
    )

    try:
        params, pcov = curve_fit(
            f=fit_func,
            xdata=x_data,
            ydata=y_data,
            p0=p0,
            bounds=bounds,
            maxfev=5000,
        )
    except Exception as e:
        raise RuntimeError(f"curve_fit failed: {e}") from e

    intercept, cross_log, slope_l, slope_h, rigidity = params

    stderr_intercept = var_estimation(value=intercept, stderr=np.sqrt(pcov[0, 0]))
    stderr_cross = var_estimation(
        value=cross_log, stderr=np.sqrt(pcov[1, 1]) * np.log(10) * 10**cross_log
    )
    stderr_slope_l = var_estimation(value=slope_l, stderr=np.sqrt(pcov[2, 2]))
    stderr_slope_h = var_estimation(value=slope_h, stderr=np.sqrt(pcov[3, 3]))
    stderr_rigidity = var_estimation(value=rigidity, stderr=np.sqrt(pcov[4, 4]))

    cross = 10**cross_log
    return params

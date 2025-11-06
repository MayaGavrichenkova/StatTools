from dataclasses import dataclass
from functools import partial

import numpy as np
from scipy.optimize import curve_fit

from StatTools.analysis.support_ff import tf


@dataclass
class var_estimation:
    value: np.ndarray
    stderr: np.ndarray


@dataclass
class ff_params:
    intercept: var_estimation
    cross: var_estimation
    slope_current: var_estimation
    ridigity: var_estimation


def cross_fcn_sloped(x, y_0, *args, crossover_amount: int):
    """
    Function which can be used as base element for fluctuation characteristic approximation with several Hurst
    coefficients with levelling ...........????????????

    Args:
      x(Union[int, Iterable]): points where  fluctuation function F(s) is calculated.
      y_0(int): y-intecept for function.
      *args: Variable length argument list.
      crossover_amount(int): value of points where the Hurst coefficient has changed.

    Returns:
      float: The return value of function with current input values.
    """
    crossovers = crossover_amount
    slopes = crossover_amount + 1
    C = args[:crossovers]
    slope = args[crossovers : crossovers + slopes]
    R = args[crossovers + slopes : crossovers + 2 * slopes]

    curr_C = None
    Ridigity = None
    slope_val = None
    prev_C = -100
    result_sloped = 0
    result = np.zeros_like(x, dtype=float)

    for index in range(slopes):
        if index < crossovers:
            curr_C = C[index]
        else:
            curr_C = 100
            pass
        slope_val = slope[index]
        Ridigity = R[index]
        result += slope_val * tf(x, Ridigity, prev_C, curr_C)
        result_sloped += slope_val * tf(0, Ridigity, prev_C, curr_C)
        prev_C = curr_C
    return y_0 + result - result_sloped


def analyse_cross_ff(
    hs: np.ndarray,
    S: np.ndarray,
    max_ridigity: float = +np.inf,
    min_ridigity: int = 1,
    max_intercept: float = +np.inf,
    min_intercept: float = -np.inf,
    min_slope_current: int = 0,
    min_slope_current_1: int = 0,
    max_slope_current_1: int = 5,
    max_slope_current: int = 5,
    ridigity_initial_parameter: int = 5,
    ridigity_initial_parameter_1: int = 5,
    intercept_initial_parameter: int = 0,
    slope_current_initial_parameter: int = 1,
    slope_current_initial_parameter_1: int = 0.8,
    min_ridigity_1: int = 1,
    max_ridigity_1: float = +np.inf,
):
    """
    Function where running fluctuation characteristic approximation with several Hurst
    coefficients. It let us receive parameters of fluctuation function after approximation and errors that can be calculated
    as diagonal elements of covariation matrix.

    Args:
      hs(array): The independent (k,M) shape-array variable where data is measured.
      S(array): The dependent data M-length array.
      max_ridigity(float): maximum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation.
      min_ridigity(int): minimum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation.
      max_intercept(float): maximum value of y-intecept for function for bounds of approximation.
      min_intercept(float): minimum value of y-intecept for function for bounds of approximation.
      min_slope_current(int): minimum value of Hurst coefficient for bounds of approximation.
      min_slope_current_1(int): minimum value of Hurst coefficient for function with several Hurst coefficients for bounds
      of approximation.
      max_slope_current(int): maximum value of Hurst coefficient for bounds of approximation.
      max_slope_current_1(int): maximum value of Hurst coefficient for function with several Hurst coefficients for bounds
      of approximation.
      ridigity_initial_parameter(int): initial value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for non-linear least squares fitting.
      ridigity_initial_parameter_1(int): initial value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for non-linear least squares fitting for function with several Hurst coefficients.
      slope_current_initial_parameter(int): initial value of Hurst coefficient for non-linear least squares fitting.
      slope_current_initial_parameter_1(int): initial value of Hurst coefficient for non-linear least squares fitting of
      function with several Hurst coefficients.
      min_ridigity_1(int): minimum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation for function with several Hurst coefficients.
      max_ridigity(float): maximum value of coefficient which is proportonal to sharpness (rigidity) of a DFA crossover
      for bounds of approximation for function with several Hurst coefficients.

    Returns:
        tuple[float, float, float, float, float, float, float, float]: [itercept, cross, slope_current, ridigity,
        intercept residuals , cross residuals, slope_current residuals, ridigity residuals], where [itercept, cross, slope_current, ridigity] - parameters
        of fluctuation function with one crossover, [intercept residuals , cross residuals, slope_current residuals, ridigity residuals] - residuals
        of parameters that can be calculated as difference between parameters of function afyter fitting and parameters of
        function with dependent data for function with one crossover.

        tuple[float, float, float, float, float, float, float, float]: [itercept, cross, slope_current, ridigity,
        intercept residuals , cross residuals, slope_current residuals, ridigity residuals], where [itercept, cross, slope_current, ridigity] - parameters
        of fluctuation function with several crossovers, [intercept residuals , cross residuals, slope_current residuals, ridigity residuals] - residuals
        of parameters that can be calculated as difference between parameters of function afyter fitting and parameters of
        function with dependent data for function with several crossovers.
    """
    hs = hs
    S = S
    s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T
    change_cross_value = partial(cross_fcn_sloped, crossover_amount=2)
    change_single_cross_value = partial(cross_fcn_sloped, crossover_amount=1)
    po = (0, np.log10(S[len(S) // 3]), np.log10(S[2 * len(S) // 3]), 1, 1, 1, 5, 5, 5)
    bounds_min = (-np.inf, np.log10(S[0]), np.log10(S[1]), 0, 0, 0, 1, 1, 1)
    bounds_max = (1, np.log10(S[-2]), np.log10(S[-1]), 5, 5, 5, 20, 20, 20)
    popt, pcov, infodict, mesg, ier = curve_fit(
        change_cross_value,
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
    intercept_err = np.array([err_y0])
    cross_values = np.array([10**C1, 10**C2])
    cross_err = np.array([err_C1, err_C2])
    slope_values = np.array([s0, s1, s2])
    slope_errs = np.array([err_s0, err_s1, err_s2])
    ridigity_values = np.array([R1, R2, R3])
    ridigity_err = np.array([err_R1, err_R2, err_R3])
    popt_single_value, pcov_single_value, infodict, mesg, ier = curve_fit(
        change_single_cross_value,
        np.log10(s.flatten()),
        np.log10(hs.flatten()),
        p0=(
            intercept_initial_parameter,
            np.log10(S[len(S) // 2]),
            slope_current_initial_parameter,
            slope_current_initial_parameter_1,
            ridigity_initial_parameter,
            ridigity_initial_parameter_1,
        ),
        bounds=(
            [
                min_intercept,
                np.log10(S[0]),
                min_slope_current,
                min_slope_current_1,
                min_ridigity,
                min_ridigity_1,
            ],
            [
                max_intercept,
                np.log10(S[-1]),
                max_slope_current,
                max_slope_current_1,
                max_ridigity,
                max_ridigity_1,
            ],
        ),
        full_output=True,
        maxfev=1000,
        nan_policy="raise",
    )
    (
        y0_single_value,
        C1_single_value,
        s0_single_value,
        s1_single_value,
        R1_single_value,
        R2_single_value,
    ) = popt_single_value
    stderr = np.sqrt(np.diag(pcov_single_value))
    (
        err_y0_single_value,
        err_C1_single_value,
        err_s0_single_value,
        err_s1_single_value,
        err_R1_single_value,
        err_R2_single_value,
    ) = stderr
    intercept_values_single = np.array([y0_single_value])
    intercept_err_single = np.array([err_y0_single_value])
    cross_values_single = np.array([10**C1_single_value])
    cross_err_single = np.array(
        [err_C1_single_value * 10**C1_single_value * np.log(10)]
    )
    slope_values_single = np.array([s0_single_value, s1_single_value])
    slope_errs_single = np.array([err_s0_single_value, err_s1_single_value])
    ridigity_values_single = np.array([R1_single_value, R2_single_value])
    ridigity_err_single = np.array([err_R1_single_value, err_R2_single_value])

    return (
        ff_params(
            intercept=var_estimation(
                value=intercept_values_single, stderr=intercept_err_single
            ),
            cross=var_estimation(
                value=cross_values_single,
                stderr=cross_err_single,
            ),
            slope_current=var_estimation(
                value=slope_values_single,
                stderr=slope_errs_single,
            ),
            ridigity=var_estimation(
                value=ridigity_values_single,
                stderr=ridigity_err_single,
            ),
        ),
        10 ** change_single_cross_value(np.log10(s), *popt_single_value) - hs,
        ff_params(
            intercept=var_estimation(value=intercept_values, stderr=intercept_err),
            cross=var_estimation(value=cross_values, stderr=cross_err),
            slope_current=var_estimation(value=slope_values, stderr=slope_errs),
            ridigity=var_estimation(value=ridigity_values, stderr=ridigity_err),
        ),
        10 ** change_cross_value(np.log10(s), *popt) - hs,
    )

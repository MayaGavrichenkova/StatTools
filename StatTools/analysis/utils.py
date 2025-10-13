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
    multiple_stderr: float


@dataclass
class dfa_params:
    intercept: var_estimation
    cross: var_estimation
    slope_l: var_estimation
    slope_h: var_estimation
    ridigity: var_estimation


@dataclass
class multiple_dfa_params:
    multiple_intercept: var_estimation_multiple
    multiple_cross: var_estimation_multiple
    multiple_slope_l: var_estimation_multiple
    multiple_slope_h: var_estimation_multiple
    multiple_ridigity: var_estimation_multiple


class Utils:

    def f_fcn(self, x, R, C):
        """
        ReLu type function with ridigity-R, C-slope scale
        """
        return np.log(1 + np.exp(R * (x - C))) / R * (x - C) / np.sqrt(1 + (x - C) ** 2)

    def rev_f_fcn(self, x, R, C):
        """
        Invert Relu type function
        """
        return (
            np.log(1 + np.exp(R * (x - C))) / R * (-(x - C)) / np.sqrt(1 + (x - C) ** 2)
        )

    def tf_minus_inf(self, x, R, C1):
        return self.rev_f_fcn(x, R, C1)

    def tf_plus_inf(self, x, R, C2):
        return self.f_fcn(x, R, C2)

    def tf(self, x, R, C1, C2):
        """
        Step function
        """
        if C1 <= float("-inf"):
            return self.tf_minus_inf(x, R, C1)
        elif C2 >= float("inf"):
            return self.tf_plus_inf(x, R, C2)
        else:
            return -self.f_fcn(x, R, C2) - self.rev_f_fcn(x, R, C1)

    def single_cross_fcn_sloped(self, x, y_0, C_12, slope_1, slope_2, R_12):

        value = None
        value = (slope_1 * self.tf(0, R_12, -100, C_12)) - (
            slope_2 * self.tf(0, R_12, C_12, 100)
        )
        func1 = slope_1 * (self.tf(x, R_12, -100, C_12))
        func2 = slope_2 * self.tf(x, R_12, C_12, 100)
        return y_0 + func1 + func2 - value

    def multiple_cross_fcn_sloped(
        self, x, y_0, C_ij: np.ndarray, slope_ij: np.ndarray, R_ij: np.ndarray
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
            result += slope * self.tf(x, R, prev_C, curr_C)
            result_sloped += slope * self.tf(0, R, prev_C, curr_C)
            prev_C = curr_C
        return y_0 + result - result_sloped

    def single_cross_fcn(self, x, y_0, C_12, slope_1, slope_2, R_12):
        """
        Creates piesewise linear function with slopes definition
        """
        return (
            y_0
            + slope_1 * self.tf(x, R_12, -100, C_12)
            + slope_2 * self.tf(x, R_12, C_12, 100)
        )

    def analyse_multiple_cross_ff(self, hs, S):
        """
        Creates piesewise linear function with multiple crossovers
        """
        hs = hs
        S = S
        s = np.repeat(S[:, np.newaxis], hs.shape[0], -1).T

        def fit_func_for_multiple_cross(
            x, multiple_cross, multiple_slope_l, multiple_slope_h, multiple_ridigity
        ):
            return self.multiple_cross_fcn_sloped(
                x, multiple_cross, multiple_slope_l, multiple_slope_h, multiple_ridigity
            )

        p0 = (0, np.log10(S[len(S) // 2]), 1, 1, 5)
        bounds = (
            [-np.inf, np.log10(S[0]), 0, 0, 1],
            [+np.inf, np.log10(S[-1]), 5, 5, +np.inf],
        )

        (
            (
                multiple_intercept,
                multiple_cross,
                multiple_slope_l,
                multiple_slope_h,
                multiple_ridigity,
            ),
            pcov,
            infodict,
            mesg,
            ier,
        ) = curve_fit(
            self.multiple_cross_fcn_sloped,
            np.log10(s.flatten()),
            np.log10(hs.flatten()),
            p0=(0, np.log10(S[len(S) // 2]), 1, 1, 5),
            bounds=(
                [-np.inf, np.log10(S[0]), 0, 0, 1],
                [+np.inf, np.log10(S[-1]), 5, 5, +np.inf],
            ),
            full_output=True,
        )

        return (
            multiple_dfa_params(
                multiple_intercept=var_estimation(
                    value=multiple_intercept, multiple_stderr=np.sqrt(pcov[0][0])
                ),
                multiple_cross=var_estimation(
                    value=10**multiple_cross, multiple_stderr=np.sqrt(pcov[1][1])
                ),
                multiple_slope_l=var_estimation(
                    value=multiple_slope_l, multiple_stderr=np.sqrt(pcov[2][2])
                ),
                multiple_slope_h=var_estimation(
                    value=multiple_slope_h, multiple_stderr=np.sqrt(pcov[3][3])
                ),
                multiple_ridigity=var_estimation(
                    value=multiple_ridigity, multiple_stderr=np.sqrt(pcov[4][4])
                ),
            ),
            10
            ** self.multiple_cross_fcn_sloped(
                np.log10(S),
                multiple_intercept,
                multiple_cross,
                multiple_slope_l,
                multiple_slope_h,
                multiple_ridigity,
            )
            - hs,
        )

    def analyse_multiple_plot_ff(
        self, hs, S, params: multiple_dfa_params, residuals: np.ndarray, title="F(S)"
    ):
        if len(residuals.shape) == 1:
            residuals = np.expand_dims(residuals, -1)
        cross = params.multiple_cross.multiple_value
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title(title)
        ax.errorbar(
            S,
            10
            ** (
                self.single_cross_fcn(
                    np.log10(S),
                    params.multiple_intercept.multiple_value,
                    np.log10(params.multiple_cross.multiple_value),
                    params.multiple_slope_l.multiple_value,
                    params.multiple_slope_h.multiple_value,
                    params.multiple_ridigity.multiple_value,
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

    def analyse_ff(self, hs, S):
        """
        Analyses real data: F(s) and s and simulated data with linear regression's model.
        """
        hs = hs
        S = S
        s = np.repeat(S[:, np.newaxis], hs.shape[0], 1).T

        def fit_func(x, cross, slope_l, slope_h, rigidity):
            """
            Сreates linear regression's model
            """
            return self.single_cross_fcn_sloped(x, cross, slope_l, slope_h, rigidity)

        p0 = (0, np.log10(S[len(S) // 2]), 1, 1, 5)
        bounds = (
            [-np.inf, np.log10(S[0]), 0, 0, 1],
            [+np.inf, np.log10(S[-1]), 5, 5, +np.inf],
        )

        (intercept, cross, slope_l, slope_h, ridigity), pcov, infodict, mesg, ier = (
            curve_fit(
                self.single_cross_fcn_sloped,
                np.log10(s.flatten()),
                np.log10(hs.flatten()),
                p0=(0, np.log10(S[len(S) // 2]), 1, 1, 5),
                bounds=(
                    [-np.inf, np.log10(S[0]), 0, 0, 1],
                    [+np.inf, np.log10(S[-1]), 5, 5, +np.inf],
                ),
                full_output=True,
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
            ** self.single_cross_fcn(
                np.log10(S), intercept, cross, slope_l, slope_h, ridigity
            )
            - hs,
        )

    def analyse_plot_ff(
        self, hs, S, params: dfa_params, residuals: np.ndarray, title="F(S)"
    ):
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
                self.single_cross_fcn(
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


# def main():
#     tst_s = np.array([0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000])
#     utils=Utils()
#     tst_cross = 3
#     tst_h = 10**utils.single_cross_fcn(np.log10(tst_s), 0, np.log10(tst_cross), 1, 0.5, 1)
#     tst_hr = 1 + np.random.normal(0, 0.3, (20, len(tst_h)))
#     tst_hr *= tst_h
#     params, residuals  = utils.analyse_ff(tst_hr, tst_s)

#     utils.analyse_plot_ff(tst_hr, tst_s, params, residuals)

# if __name__=="__main__":
#       main()

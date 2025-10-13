import matplotlib.pyplot as plt
import numpy as np

from StatTools.analysis import Utils


def plot_single_cross_result():
    utils = Utils()
    intercept = 0
    C_12 = 4
    slope_1 = 2
    slope_2_1 = 1
    slope_2_2 = 4
    R_1 = 5
    R_2 = 1
    x = np.linspace(0, 50, num=101, endpoint=True)
    y_1 = utils.single_cross_fcn_sloped(x, intercept, C_12, slope_1, slope_2_1, R_1)
    y_2 = utils.single_cross_fcn_sloped(x, intercept, C_12, slope_1, slope_2_2, R_2)

    fig, ax = plt.subplots()
    plt.plot(x, y_1, label="$R_{12} = 5$")
    plt.plot(x, y_2, label="$R_{12} = 1$")
    plt.grid(which="both")
    plt.legend()
    plt.show()


def plot_multiple_cross_result():
    utils = Utils()
    t = np.linspace(0, 45, num=101, endpoint=True)
    y = np.linspace(-20, 100, num=101, endpoint=True)
    slope_ij = [1, 6, 3, 1, 6]
    C_ij = [5, 6, 15, 20]
    R_ij = [5, 7, 1, 2, 1]
    fig, ax = plt.subplots()
    fig.set_size_inches(13, 5)
    plt.subplot(1, 1, 1)
    plt.axhline(y=0, color="r", linestyle="--", label="y0")
    plt.plot(
        t,
        utils.multiple_cross_fcn_sloped(t, 0, C_ij, slope_ij, R_ij),
        label="sum of func sloped",
    )
    plt.grid()
    plt.legend()
    plt.xlim(0, 45)
    plt.show()


plot_single_cross_result()
plot_multiple_cross_result()

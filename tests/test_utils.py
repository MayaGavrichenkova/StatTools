import numpy as np

from StatTools.analysis import Utils


def main():
    tst_s = np.array(
        [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
    )
    utils = Utils()
    tst_cross = 3
    tst_h = 10 ** utils.single_cross_fcn(
        np.log10(tst_s), 0, np.log10(tst_cross), 1, 0.5, 1
    )
    tst_hr = 1 + np.random.normal(0, 0.3, (20, len(tst_h)))
    tst_hr *= tst_h
    params, residuals = utils.analyse_ff(tst_hr, tst_s)

    utils.analyse_plot_ff(tst_hr, tst_s, params, residuals)


if __name__ == "__main__":
    main()

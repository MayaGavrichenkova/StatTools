import numpy as np

from StatTools.analysis.utils import analyse_cross_ff, cross_fcn_sloped
from StatTools.visualization.ff_plot import plot_ff

tst_s = np.array(
    [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
)

y_0 = 0

cross_single = [5]
slope_single = [1, 4]
r_single = [2, 3]
cross = [5, 25]
slope = [1, 4, 2]
r = [2, 3, 1]

tst_h_multiple = 10 ** cross_fcn_sloped(
    np.log10(tst_s),
    y_0,
    np.log10(cross[0]),
    np.log10(cross[1]),
    slope[0],
    slope[1],
    slope[2],
    r[0],
    r[1],
    r[2],
    crossover_amount=2,
)
tst_hr_multiple = 1 + np.random.lognormal(0, 0.3, (20, len(tst_h_multiple)))
tst_hr_multiple *= tst_h_multiple
_, _, _, residuals = analyse_cross_ff(tst_hr_multiple, tst_s)
plot_ff(tst_hr_multiple, tst_s, residuals, y_0, r, cross, slope, crossover_amount=2)

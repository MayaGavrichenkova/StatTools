import numpy as np
import pytest
from numpy.testing import assert_allclose

from StatTools.analysis.utils import (
    analyse_multiple_cross_ff,
    multiple_cross_fcn_sloped,
)


def test_multiple_crossovers_utils():
    slope_ij = [1, 4, 3]
    C_ij = [5, 6]
    R_ij = [5, 7, 1]
    tst_s = np.array(
        [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
    )
    ff = 10 ** multiple_cross_fcn_sloped(
        np.log10(tst_s), 0, np.log10(C_ij), slope_ij, R_ij
    )
    tst_hr = 1 + np.random.normal(0, 0.3, (20, len(ff)))
    tst_hr *= ff
    ff_multiple_params, _ = analyse_multiple_cross_ff(tst_hr, tst_s)
    assert_allclose(
        ff_multiple_params.multiple_slope_current.multiple_value,
        slope_ij,
        rtol=0.1,
        atol=0.3,
    )

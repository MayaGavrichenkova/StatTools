import numpy as np
import pytest
from scipy import stats

from StatTools.analysis.utils import analyse_ff, single_cross_fcn


def test_two_slopes_utils():

    h1, h2 = 1, 2
    cross_theor = 10

    tst_s = np.array(
        [0.01, 0.1, 0.3, 0.5, 1, 1.5, 2.5, 5, 7.5, 10, 15, 20, 50, 100, 250, 500, 1000]
    )
    ff = 10 ** single_cross_fcn(np.log10(tst_s), 0, np.log10(cross_theor), h1, h2, 5)
    tst_hr = 1 + np.random.normal(0, 0.3, (20, len(ff)))
    tst_hr *= ff
    ff_params, _ = analyse_ff(tst_hr, tst_s)
    assert ff_params.slope_l.value == pytest.approx(h1, abs=0.2)
    assert ff_params.slope_h.value == pytest.approx(h2, abs=0.2)
    assert ff_params.cross.value == pytest.approx(cross_theor, rel=0.3)

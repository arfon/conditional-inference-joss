import numpy as np
import pytest
from scipy.stats import norm

from conditional_inference.significance_condition import SignificanceCondition


def test_common_methods():
    model = SignificanceCondition(np.arange(4), np.identity(4))
    results = model.fit()
    results.conf_int(columns=["x3"])
    results.summary(columns=["x3"])
    results.point_plot(columns=["x3"])


class TestSignificanceCondition:
    def test_get_marginal_distribution(self):
        dist = SignificanceCondition(4, 4).get_marginal_distribution(0)
        assert dist.truncnorm_kwargs["scale"] == 2
        np.testing.assert_array_almost_equal(
            dist.truncnorm_kwargs["truncation_set"],
            [[-np.inf, -2 * norm.ppf(0.975)], [2 * norm.ppf(0.975), np.inf]],
        )

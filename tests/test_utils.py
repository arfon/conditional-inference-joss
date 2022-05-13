from turtle import home
from unittest.mock import Mock

import numpy as np
from scipy.stats import multivariate_normal

from conditional_inference.utils import (
    expected_wasserstein_distance,
    holm_bonferroni_correction,
    weighted_quantile,
)


def test_expected_wasserstein_distance():
    # expected Wasserstein distance should be smaller when the parameters are estimated with greater precision
    n_params = 3
    mean, cov = np.arange(n_params), np.identity(n_params)
    rvs0 = multivariate_normal.rvs(mean, 1, size=100)
    rvs1 = multivariate_normal.rvs(mean, 0.1 ** 2, size=100)
    assert expected_wasserstein_distance(
        mean, cov, rvs0
    ) > expected_wasserstein_distance(mean, cov, rvs1)


def test_holm_bonferroni_correction():
    results = Mock()
    results.pvalues = np.array([0.1, 0.05, 0.01])
    results.model = Mock()
    results.model.exog_names = np.array(["x0", "x1", "x2"])
    correction = holm_bonferroni_correction(results=results)
    np.testing.assert_array_equal(correction.pvalues, [0.01, 0.05, 0.1])
    np.testing.assert_array_equal(correction.significant, [True, False, False])
    np.testing.assert_array_equal(correction.index, ["x2", "x1", "x0"])


def test_weighted_quantile():
    quantiles = [0, 0.25, 0.5, 0.75, 1]
    np.testing.assert_array_almost_equal(
        weighted_quantile(np.linspace(0, 1), quantiles), quantiles, decimal=2
    )

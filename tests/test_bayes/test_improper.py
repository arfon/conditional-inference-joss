import numpy as np
import pytest
from scipy.stats import norm

from conditional_inference.bayes import Improper

from .utils import run_common_methods

mean, cov = np.arange(3), np.identity(3)
model = Improper(mean, cov)
results = model.fit()


def test_common_methods():
    run_common_methods(results)


@pytest.mark.parametrize("index", (0, 1, 2))
def test_get_marginal_distribution(index):
    dist = model.get_marginal_distribution(index)
    assert dist.mean() == mean[index]
    assert dist.var() == cov[index, index]


def test_get_joint_distribution():
    dist = model.get_joint_distribution()
    np.testing.assert_almost_equal(dist.mean, mean)
    np.testing.assert_almost_equal(dist.cov, cov)


def test_conf_int():
    conf_int = results.conf_int()
    norm_conf_int = norm.ppf([0.025, 0.975])
    norm_conf_int = np.array([norm_conf_int, norm_conf_int + 1, norm_conf_int + 2])
    np.testing.assert_array_almost_equal(conf_int, norm_conf_int)

from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm, truncnorm as scipy_truncnorm

from conditional_inference.stats import quantile_unbiased, truncnorm


VALUES = np.linspace(-2, 2, num=5)
LOC = [-1, 0, 1]
SCALE = [1, 2]
TRUNCATION_SET = [
    (-np.inf, -1),
    (-1, 1),
    (1, np.inf)
]


@pytest.fixture(scope="module", params=list(product(LOC, SCALE, TRUNCATION_SET)))
def truncnorm_distributions(request):
    loc, scale, truncation_set = request.param
    return (
        truncnorm([truncation_set], loc=loc, scale=scale),
        scipy_truncnorm(*truncation_set, loc=loc, scale=scale)
    )


class TestTruncnorm:
    # test that conditional inference truncnorm behaves like scipy truncnorm
    # for reasonable ranges of values
    # conditional inference truncnorm should perform better in tails
    def test_pdf(self, truncnorm_distributions):
        dist0, dist1 = truncnorm_distributions
        assert_allclose(dist0.pdf(VALUES), dist1.pdf(VALUES), atol=1e-3)

    def test_logpdf(self, truncnorm_distributions):
        dist0, dist1 = truncnorm_distributions
        assert_allclose(dist0.logpdf(VALUES), dist1.logpdf(VALUES), atol=1e-3)

    def test_cdf(self, truncnorm_distributions):
        dist0, dist1 = truncnorm_distributions
        assert_allclose(dist0.cdf(VALUES), dist1.cdf(VALUES), atol=1e-3)

    def test_logcdf(self, truncnorm_distributions):
        dist0, dist1 = truncnorm_distributions
        assert_allclose(dist0.logcdf(VALUES), dist1.logcdf(VALUES), atol=1e-3)

    def test_tails(self):
        # test that truncnorm can handle extreme truncation sets
        assert truncnorm([(8, np.inf)]).cdf(8.5) < 1
        assert truncnorm([(-np.inf, -8)]).cdf(-8.5) > 0
        assert truncnorm([(100, np.inf)]).cdf(101) <= 1
        assert truncnorm([(-np.inf, -100)]).cdf(-101) >= 0

    def test_default_truncation_set(self):
        assert_allclose(truncnorm().ppf([.25, .5, .75]), norm().ppf([.25, .5, .75]))

    def test_concave_truncation_set(self):
        truncnorm([(-2, -1), (1, 2)]).ppf([.05, .25, .5, .75, .95])


@pytest.fixture(scope="module", params=list(product(LOC, SCALE, TRUNCATION_SET)))
def quantile_unbiased_distribution(request):
    loc, scale, truncation_set = request.param
    return quantile_unbiased(loc, scale=scale, truncation_set=[truncation_set])


class TestQuantileUnbiased:
    def test_pdf(self, quantile_unbiased_distribution):
        quantile_unbiased_distribution.pdf(VALUES)

    # def test_cdf(self, quantile_unbiased_distribution):
    #     quantile_unbiased_distribution.cdf(VALUES)

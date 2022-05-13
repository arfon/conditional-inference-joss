from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm, truncnorm as scipy_truncnorm

from conditional_inference.stats import (
    joint_distribution,
    mixture,
    nonparametric,
    quantile_unbiased,
    truncnorm,
)


VALUES = np.linspace(-2, 2, num=5)
LOC = [-1, 0, 1]
SCALE = [1, 2]
TRUNCATION_SET = [(-np.inf, -1), (-1, 1), (1, np.inf)]


class TestJointDistribution:
    marginals = [norm(), norm(4)]
    dist = joint_distribution(marginals)
    values = np.vstack([marginals[0].rvs(10), marginals[1].rvs(10)]).T

    def test_logpdf(self):
        np.testing.assert_array_almost_equal(
            self.dist.logpdf(self.values),
            self.marginals[0].logpdf(self.values[:, 0])
            + self.marginals[1].logpdf(self.values[:, 1]),
        )

    def test_pdf(self):
        np.testing.assert_array_almost_equal(
            self.dist.pdf(self.values),
            self.marginals[0].pdf(self.values[:, 0])
            * self.marginals[1].pdf(self.values[:, 1]),
        )

    @pytest.mark.parametrize("size", (1, 10))
    def test_rvs(self, size):
        assert self.dist.rvs(size).shape == (size, 2)


class TestMixture:
    mixed = [norm(), norm(4)]
    dist = mixture(mixed)

    def test_pdf(self):
        np.testing.assert_array_almost_equal(
            self.dist.pdf(VALUES),
            0.5 * (self.mixed[0].pdf(VALUES) + self.mixed[1].pdf(VALUES)),
        )

    def test_cdf(self):
        np.testing.assert_array_almost_equal(
            self.dist.cdf(VALUES),
            0.5 * (self.mixed[0].cdf(VALUES) + self.mixed[1].cdf(VALUES)),
        )

    def test_mean(self):
        assert self.dist.mean() == 0.5 * (self.mixed[0].mean() + self.mixed[1].mean())

    def test_variance(self):
        assert self.dist.var() == 0.5 * (self.mixed[0].var() + self.mixed[1].var())


class TestNonparametric:
    x = np.linspace(-3, 3)
    dist = nonparametric((x, norm.pdf(x)))

    def test_pdf(self):
        np.testing.assert_array_almost_equal(
            self.dist.pdf(self.x), norm.pdf(self.x), decimal=2
        )

    def test_cdf(self):
        np.testing.assert_array_almost_equal(
            self.dist.cdf(self.x), norm.cdf(self.x), decimal=1
        )

    def test_ppf(self):
        q = np.linspace(0.025, 0.975, num=10)
        np.testing.assert_array_almost_equal(self.dist.ppf(q), norm.ppf(q), decimal=2)

    def test_mean(self):
        assert abs(self.dist.mean() - norm.mean()) < 0.01

    def test_std(self):
        assert abs(self.dist.std() - norm.std()) < 0.02


@pytest.fixture(scope="module", params=list(product(LOC, SCALE, TRUNCATION_SET)))
def quantile_unbiased_distribution(request):
    loc, scale, truncation_set = request.param
    return quantile_unbiased(loc, scale=scale, truncation_set=[truncation_set])


class TestQuantileUnbiased:
    # the quantile unbiased distribution behaves like a normal when the truncation set
    # is all real values
    untruncated_dist = quantile_unbiased(0, scale=1, truncation_set=[(-np.inf, np.inf)])

    def test_pdf(self, quantile_unbiased_distribution):
        quantile_unbiased_distribution.pdf(VALUES)

    def test_untruncated_pdf(self):
        x = np.linspace(-2, 2)
        np.testing.assert_array_almost_equal(self.untruncated_dist.pdf(x), norm.pdf(x))

    def test_cdf(self, quantile_unbiased_distribution):
        quantile_unbiased_distribution.cdf(VALUES)

    def test_untruncated_cdf(self):
        x = np.linspace(-2, 2)
        np.testing.assert_array_almost_equal(self.untruncated_dist.cdf(x), norm.cdf(x))

    def test_ppf(self, quantile_unbiased_distribution):
        quantile_unbiased_distribution.ppf(np.linspace(0, 1, 5))

    def test_untruncated_ppf(self):
        x = np.linspace(0.025, 0.975, 5)
        np.testing.assert_almost_equal(self.untruncated_dist.ppf(x), norm.ppf(x))


@pytest.fixture(scope="module", params=list(product(LOC, SCALE, TRUNCATION_SET)))
def truncnorm_distributions(request):
    loc, scale, truncation_set = request.param
    return (
        truncnorm([truncation_set], loc=loc, scale=scale),
        scipy_truncnorm(*truncation_set, loc=loc, scale=scale),
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
        assert_allclose(
            truncnorm().ppf([0.25, 0.5, 0.75]), norm().ppf([0.25, 0.5, 0.75])
        )

    def test_concave_truncation_set(self):
        truncnorm([(-2, -1), (1, 2)]).ppf([0.05, 0.25, 0.5, 0.75, 0.95])

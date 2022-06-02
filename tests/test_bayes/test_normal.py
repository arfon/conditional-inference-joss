from itertools import product

import numpy as np
import pytest
from scipy.stats import norm

from conditional_inference.bayes import Normal

from .utils import run_common_methods

n_params = 20
mean, cov = np.arange(n_params), np.identity(n_params)
X = np.vstack([np.ones(n_params), int(n_params / 2) * [0] + int(n_params / 2) * [1]]).T
mean = mean - mean.mean()


@pytest.fixture(
    scope="module", params=product((None, X), ("mle", "bock", "james_stein"))
)
def model(request):
    X, fit_method = request.param
    return Normal(mean, cov, X, fit_method=fit_method)


@pytest.fixture(scope="module")
def results(model):
    return model.fit()


def test_common_methods(results):
    run_common_methods(results)


def get_expected_means(model):
    if model.X.shape[1] == 1:
        return 0, 0

    return mean[: int(n_params / 2)].mean(), mean[int(n_params / 2) :].mean()


def test_get_marginal_prior(model, atol=0.01):
    expected_mean_0, expected_mean_1 = get_expected_means(model)
    dist_0 = model.get_marginal_prior(0)
    assert abs(dist_0.mean() - expected_mean_0) < atol
    dist_1 = model.get_marginal_prior(int(n_params / 2))
    assert abs(dist_1.mean() - expected_mean_1) < atol


def test_get_marginal_distribution(model):
    posterior_mean = np.array(
        [model.get_marginal_distribution(i).mean() for i in range(n_params)]
    )
    assert (np.diff(posterior_mean) > 0).all()

    # tests shrinkage
    expected_mean_0, expected_mean_1 = get_expected_means(model)
    expected_means = np.array(
        int(n_params / 2) * [expected_mean_0] + int(n_params / 2) * [expected_mean_1]
    )
    np.testing.assert_array_equal(mean < posterior_mean, mean < expected_means)


def test_conf_int(results):
    conf_int = results.conf_int()
    norm_len_ci = np.diff(norm.ppf([0.025, 0.975]))
    # test that Bayesian CIs are shorter than conventional CIs
    assert (np.diff(conf_int, axis=1) < norm_len_ci).all()


def test_bock():
    # can compute Bock's Stein-type estimates analytically
    cov = np.identity(n_params)
    results = Normal(mean, cov, prior_mean=0, fit_method="bock").fit()
    expected_result = (
        1
        - (np.trace(cov) / np.linalg.eig(cov)[0].max() - 2)
        / (mean.reshape(1, -1) @ np.linalg.inv(cov) @ mean.reshape(-1, 1))
    ) * mean
    np.testing.assert_array_almost_equal(results.params, expected_result.squeeze())


def test_james_stein():
    # can compute the James-Stein estimates analytically
    results = Normal(mean, cov, prior_mean=0, fit_method="james_stein").fit()
    np.testing.assert_array_almost_equal(
        results.params,
        (1 - (n_params - 2) * np.sqrt(cov[0, 0]) / (mean ** 2).sum()) * mean,
    )


@pytest.mark.parametrize("fit_method", ("mle", "james_stein"))
def test_zero_prior_cov(fit_method):
    # make sure the models are robust when there is 0 prior covariance
    mean = np.array(
        [
            -3.39359413e-01,
            -6.62381513e-01,
            -1.51536892e-01,
            -2.58385772e-01,
            6.10271496e-01,
            8.87349385e-01,
            3.16365311e-01,
            2.27194076e-03,
            -1.26127278e00,
            -1.05872185e-01,
            4.13153327e-03,
            -5.08449489e-01,
            5.78252783e-01,
            -1.48959519e-01,
            4.96132247e-01,
            -2.48655048e00,
            -8.59522707e-01,
            -1.07444613e00,
            2.26257613e-01,
            -8.14765943e-01,
        ]
    )
    model = Normal(mean, np.identity(len(mean)), fit_method=fit_method)
    results = model.fit()
    np.testing.assert_array_almost_equal(results.params, mean.mean())

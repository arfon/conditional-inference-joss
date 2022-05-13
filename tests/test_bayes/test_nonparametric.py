import numpy as np
import pytest
from scipy.stats import norm

from conditional_inference.bayes import Nonparametric

from .utils import run_common_methods

n_params = 20
mean, cov = np.arange(n_params), np.identity(n_params)
X = np.vstack([np.ones(n_params), int(n_params / 2) * [0] + int(n_params / 2) * [1]]).T
mean = mean - mean.mean()


@pytest.fixture(scope="module", params=(None, X))
def model(request):
    X = request.param
    n_clusters = 1 if X is None else 2
    return Nonparametric(mean, cov, X=X, n_clusters=n_clusters)


@pytest.fixture(scope="module")
def results(model):
    return model.fit()


def test_common_methods(results):
    run_common_methods(results)


def get_expected_means(model):
    if model.X.shape[1] == 1:
        return 0, 0

    return mean[: int(n_params / 2)].mean(), mean[int(n_params / 2) :].mean()


def test_get_marginal_prior(model, atol=0.5):
    # with so few observations, the nonparametric prior can't fit precisely, so we need
    # a high tolerance for error
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
    if model.X.shape[1] == 1:
        indices = [0, -1]
        expected_means = [expected_mean_0, expected_mean_1]
    else:
        indices = [0, int(n_params / 2) - 1, int(n_params / 2), -1]
        expected_means = 2 * [expected_mean_0] + 2 * [expected_mean_1]

    posterior_mean = posterior_mean[indices]
    np.testing.assert_array_equal(
        mean[indices] < posterior_mean, mean[indices] < expected_means
    )


def test_conf_int(results):
    conf_int = results.conf_int()
    norm_len_ci = np.diff(norm.ppf([0.025, 0.975]))
    # test that Bayesian CIs are shorter than conventional CIs
    indices = (
        [0, -1]
        if results.model.X.shape[1] == 1
        else [0, int(n_params / 2) - 1, int(n_params / 2), -1]
    )
    assert (np.diff(conf_int, axis=1)[indices] < norm_len_ci).all()

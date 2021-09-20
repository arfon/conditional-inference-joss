import numpy as np
import pytest
from scipy.stats import loguniform

from conditional_inference.bayes.classic import LinearClassicBayes
from conditional_inference.bayes.empirical import LinearEmpiricalBayes, JamesStein
from conditional_inference.bayes.hierarchical import LinearHierarchicalBayes

atol = .1
n_policies = 4
mean = np.arange(n_policies)
cov = np.identity(n_policies)

_, prior_std_anchor = LinearEmpiricalBayes(mean, cov).estimate_prior_params()
hyperprior = loguniform(.5 * prior_std_anchor, 2 * prior_std_anchor)
# tuples of (model class, constructor keyword arguments, fit keyword arguments)
models = [
    (LinearClassicBayes, dict(prior_cov=0), {}),
    (LinearClassicBayes, dict(prior_cov=np.inf), {}),
    (LinearEmpiricalBayes, {}, {}),
    (
        LinearEmpiricalBayes,
        {},
        dict(estimate_prior_params_kwargs=dict(method="wasserstein", n_samples=20))
    ),
    (JamesStein, {}, {}),
    (LinearHierarchicalBayes, dict(prior_cov_params_distribution=hyperprior), {}),
]


@pytest.fixture(scope="module", params=models)
def results(request):
    model_cls, init_kwargs, fit_kwargs = request.param
    return model_cls(mean, cov, **init_kwargs).fit(**fit_kwargs)


class TestResults:
    def test_conf_int(self, results):
        results.conf_int()

    def test_point_plot(self, results):
        results.point_plot()

    def test_summary(self, results):
        results.summary()

    def test_wasserstein(self, results):
        distance0 = results.expected_wasserstein_distance()
        distance1 = results.expected_wasserstein_distance(mean, cov)
        assert abs(distance0 - distance1) <= atol

    def test_likelihood(self, results):
        assert results.likelihood() == results.likelihood(mean, cov)
    
    def test_rank_matrix(self, results):
        results.rank_matrix_plot()

    def test_reconstruction_histogram(self, results):
        results.reconstruction_histogram()

    def test_reconstruction_point_plot(self, results):
        results.reconstruction_point_plot()


def test_prior_mean_rvs(size=10):
    # TODO: test with estimate_prior_params keyword arguments
    model = LinearEmpiricalBayes(mean, cov)
    assert model.prior_mean_rvs().shape == (n_policies,)
    assert model.prior_mean_rvs(size).shape == (n_policies, size)

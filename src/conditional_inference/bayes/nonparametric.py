"""Nonparametric empirical Bayes.

References:

    .. code-block::

        @article{cai2021nonparametric,
            title={Nonparametric empirical bayes estimation and testing for sparse and heteroscedastic signals},
            author={Cai, Junhui and Han, Xu and Ritov, Ya'acov and Zhao, Linda},
            journal={arXiv preprint arXiv:2106.08881},
            year={2021}
        }

Notes:

    This implementation is based on Cai et al.'s nonparametric Dirac delta prior. Future
    work should also implement their mixture model with a Laplace prior.
"""
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import loguniform, norm, rv_continuous
from sklearn.cluster import KMeans
from sklearn.model_selection import check_cv
from sklearn.neighbors import KernelDensity

from ..stats import mixture, nonparametric
from .base import BayesBase


class Nonparametric(BayesBase):
    """Bayesian model with a nonparametric Dirac delta prior.

    Args:
        num (int, optional): Number of parameters to fit for the prior. Defaults to 100.
        n_clusters (int, optional): Number of clusters to use for featurized
            estimation. Defaults to 1.
        cv (int, optional): Determines the cross validation splitting strategy (input to
            ``sklearn.model_selection.check_cv``). Defaults to 5.
        rtol (float, optional): Relative tolerance stopping criteria for expectation
            maximization. The EM algorithm terminates when the relative improvement
            between iterations falls below this threshold. Defaults to .99.
        max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
        bandwidth_rvs_size (int, optional): Number of bandwidth values to try when
            tuning the kernel density estimator in between EM iterations to smooth the
            prior. Defaults to 32.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.bayes import Nonparametric

            np.random.seed(0)

            model = Nonparametric(np.arange(10), np.identity(10))
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                       Bayesian estimates
            =======================================
                coef pvalue (1-sided) [0.025 0.975]
            ---------------------------------------
            x0 0.686            0.197 -0.594  2.148
            x1 1.226            0.062 -0.189  2.965
            x2 1.977            0.011  0.304  3.982
            x3 2.982            0.001  0.991  4.941
            x4 3.990            0.000  1.960  5.892
            x5 4.943            0.000  3.050  7.061
            x6 6.036            0.000  3.989  8.037
            x7 7.063            0.000  5.024  8.673
            x8 7.768            0.000  6.089  9.123
            x9 8.264            0.000  6.862  9.506
            ===============
            Dep. Variable y
            ---------------
    """

    def __init__(
        self,
        *args: Any,
        num: int = 100,
        n_clusters: int = 1,
        cv=5,
        rtol: float = 0.99,
        max_iter: int = 100,
        bandwidth_rvs_size: int = 32,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        std = self.mean.std()
        lower, upper = self.mean.min() - 2 * std, self.mean.max() + 2 * std
        # (num,) array of values over which the prior is defined
        self._values = np.linspace(lower, upper, num)
        # (num, n_clusters) probability mass function
        self._pmf_values = np.full((num, n_clusters), 1 / num)
        # (# params, n_clusters) mixture weights for each parameter
        self._mixture_weights = KMeans(n_clusters).fit_transform(self.X)
        if (self._mixture_weights == 0).all():
            self._mixture_weights = np.ones(self._mixture_weights.shape)
        self._mixture_weights = (
            self._mixture_weights.T / self._mixture_weights.sum(axis=1)
        ).T

        def loss(value, index, cluster):
            factor = (1 - value) / (1 - self._pmf_values[index, cluster])
            self._pmf_values[:, cluster] *= factor
            self._pmf_values[index, cluster] = value
            arr = self._mixture_weights * (conditional_pdf @ self._pmf_values)
            return -np.log(arr.sum(axis=1)).sum()

        # density function of the conventional estimates evaluated at self._values
        conditional_pdf = [
            norm.pdf(self._values, mean_i, np.sqrt(variance_i))
            for mean_i, variance_i in zip(self.mean, self.cov.diagonal())
        ]
        conditional_pdf = np.array(conditional_pdf)
        # fit the prior using an EM algorithm
        prev_loss, current_loss, i = np.inf, None, 0
        values = self._values.reshape(-1, 1)
        index_cluster = list(product(np.arange(num), np.arange(n_clusters)))
        index_cluster = np.array(index_cluster).astype(int)
        cv = check_cv(cv)
        cv.shuffle = True
        for i in range(max_iter):
            # optimize each value of ``self._pmf_values``
            np.random.shuffle(index_cluster)
            for index, cluster in index_cluster:
                current_loss = minimize_scalar(
                    loss, bounds=(0, 1), method="bounded", args=(index, cluster)
                ).fun

            # smooth the PMF using a kernel density estimator
            cv.random_state = i
            for cluster in range(n_clusters):
                pmf_values = self._pmf_values[:, cluster]
                mean = np.average(self._values, weights=pmf_values)
                std = np.sqrt(
                    np.average((self._values - mean) ** 2, weights=pmf_values)
                )
                bandwidth_rvs = loguniform(0.1 * std, 2 * std).rvs(bandwidth_rvs_size)
                best_score = -np.inf
                for bandwidth in bandwidth_rvs:
                    for train_index, test_index in cv.split(values):
                        X_train, X_test = values[train_index], values[test_index]
                        weight_train = pmf_values[train_index]
                        weight_test = pmf_values[test_index]
                        weight_train /= weight_train.sum()
                        weight_test /= weight_test.sum()
                        kde = KernelDensity(bandwidth=bandwidth).fit(
                            X_train, sample_weight=weight_train
                        )
                        score = (weight_test * kde.score_samples(X_test)).mean()
                        if score > best_score:
                            best_score, best_bandwidth = score, bandwidth

                kde = KernelDensity(bandwidth=best_bandwidth).fit(
                    values, sample_weight=pmf_values
                )
                self._pmf_values[:, cluster] = np.exp(kde.score_samples(values))

            self._pmf_values /= self._pmf_values.sum(axis=0)
            if current_loss / prev_loss > rtol:
                break
            prev_loss = current_loss

        # fit a nonparametric distribution for each cluster
        self._cluster_distributions = [
            nonparametric((self._values, self._pmf_values[:, i]))
            for i in range(n_clusters)
        ]

    def _get_marginal_prior(self, index: int) -> rv_continuous:
        if len(self._cluster_distributions) == 1:
            return self._cluster_distributions[0]

        return mixture(self._cluster_distributions, self._mixture_weights[index])

    def _get_marginal_distribution(self, index: int) -> rv_continuous:
        pmf = (self._pmf_values * self._mixture_weights[index]).sum(axis=1)
        logpmf = np.log(pmf) + norm.logpdf(
            self._values, self.mean[index], np.sqrt(self.cov[index, index])
        )
        return nonparametric((self._values, np.exp(logpmf - logpmf.max())))

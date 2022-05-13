"""Bayesian model with an improper prior.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import multivariate_normal, norm, rv_continuous

from .base import BayesBase


class Improper(BayesBase):
    """Bayesian model with an improper prior.

    The improper prior is a uniform distribution on $(-\infty, \infty)$. The posterior
    is equivalent to the conventionally estimated joint normal distribution.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.bayes import Improper

            model = Improper(np.arange(10), np.identity(10))
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                       Bayesian estimates
            =======================================
                coef pvalue (1-sided) [0.025 0.975]
            ---------------------------------------
            x0 0.000            0.500 -1.960  1.960
            x1 1.000            0.159 -0.960  2.960
            x2 2.000            0.023  0.040  3.960
            x3 3.000            0.001  1.040  4.960
            x4 4.000            0.000  2.040  5.960
            x5 5.000            0.000  3.040  6.960
            x6 6.000            0.000  4.040  7.960
            x7 7.000            0.000  5.040  8.960
            x8 8.000            0.000  6.040  9.960
            x9 9.000            0.000  7.040 10.960
            ===============
            Dep. Variable y
            ---------------
    """

    def _get_marginal_prior(self, index: int) -> rv_continuous:
        raise RuntimeError(
            "The improper prior is a uniform distribution from -inf to inf"
        )

    def _get_marginal_distribution(self, index: int) -> rv_continuous:
        return norm(self.mean[index], np.sqrt(self.cov[index, index]))

    def _get_joint_prior(self, indices: np.ndarray):
        raise RuntimeError(
            "The improper prior is a uniform distribution from -inf to inf"
        )

    def _get_joint_distribution(self, indices: np.ndarray):
        return multivariate_normal(self.mean[indices], self.cov[indices][:, indices])

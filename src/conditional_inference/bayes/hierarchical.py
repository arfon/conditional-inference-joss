"""Hierarchical Bayesian analysis
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union
from typing_extensions import Protocol

import numpy as np
from scipy.stats import multivariate_normal

from ..base import ColumnsType, Numeric1DArray
from ..utils import weighted_cdf, weighted_quantile
from .base import BayesModelBase, BayesResults

PriorParams = Union[float, np.ndarray]


class Distribution(Protocol):
    def rvs(self, size=None, random_state=None):
        ...


class HierarchicalBayesBase(BayesModelBase):
    """Mixin for hierarchical Bayesian models.

    Inherits from :class:`conditional_inference.bayes.base.BayesModelBase`.

    Assumes a known distribution of prior covariance parameters.

    Args:
        mean (Numeric1DArray): (n,) array of conventionally-estimated means.
        cov (np.ndarray): (n, n) covariance matrix.
        prior_cov_params_distribution (Distribution): Distribution of prior covariance
            parameters. Must implement an `rvs` method.
        X (np.ndarray, optional): (n, p) feature matrix. Defaults to None.
        *args (Any): Passed to ``BayesModelBase``.
        **kwargs (Any): Passed to ``BayesModelBase``.

    Attributes:
        prior_cov_params_distribution (Distribution): Distribution of prior covariance
            parameters.
    """

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        prior_cov_params_distribution: Distribution,
        *args: Any,
        X: np.ndarray = None,
        **kwargs: Any,
    ):
        super().__init__(mean, cov, *args, X=X, **kwargs)
        self.prior_cov_params_distribution = prior_cov_params_distribution

    def estimate_prior_cov(self, prior_cov_params: PriorParams) -> np.ndarray:
        """Estimate the prior covariance matrix.

        Args:
            prior_cov_params (PriorParams): Parameters which determine the prior
                covariance matrix.

        Raises:
            NotImplementedError: Classes which inherit the mixin should implement this
                method.

        Returns:
            np.ndarray: (n, n) prior covariance matrix.
        """
        raise NotImplementedError()  # pragma: no cover

    def prior_mean_rvs(self, prior_cov: np.ndarray, size: int = 1) -> np.ndarray:
        """Sample means from distribution of prior means.

        Args:
            prior_cov (np.ndarray): (n, n) prior covariance matrix.
            size (int, optional): Number of samples to draw. Defaults to 1.

        Raises:
            NotImplementedError: Classes which inherit the mixin should implement this
                method.

        Returns:
            np.ndarray: (size, n) array of prior mean samples.
        """
        raise NotImplementedError()  # pragma: no cover

    def fit(
        self,
        cols: ColumnsType = None,
        n_samples: int = 1000,
        title: str = "Hierarchical Bayes estimates",
    ) -> HierarchicalBayesResults:
        """Fit the empirical Bayes estimator and return results.

        Args:
            cols (ColumnsType, optional): Names or indices of the policies of interest.
                Defaults to None.
            n_samples (int, optional): Number of samples used to approximate posterior
                distributions. Defualts to 1000.
            title (str, optional): Results title. Defaults to
                "Hierarchical Bayes results".

        Returns:
            HierarchicalResults: Hierarchical Bayes estimation results.
        """
        posterior_mean_rvs, sample_weight = self.posterior_mean_rvs(size=n_samples)
        return HierarchicalBayesResults(
            self, cols, posterior_mean_rvs, sample_weight, title=title
        )

    def prior_cov_rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Sample covariance matrices from distribution of prior covariances.

        Args:
            size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: (size, n, n) matrix of sampled prior covariances, (size,) array
                of sample weights.
        """
        prior_cov_params_sample = self.prior_cov_params_distribution.rvs(size)
        prior_covs, log_likelihood = [], []

        for prior_cov_params in prior_cov_params_sample:
            prior_cov = self.estimate_prior_cov(prior_cov_params)
            prior_covs.append(prior_cov)
            log_likelihood.append(self._scaled_log_likelihood(prior_cov))

        likelihood = np.exp(log_likelihood - np.array(log_likelihood).max())
        return np.array(prior_covs), likelihood / likelihood.sum()

    def posterior_mean_rvs(self, size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Sample mean vectors from distribution of posterior means.

        Args:
            size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: (size, n) matrix of sampled posterior mean vectors.
        """
        prior_covs, sample_weight = self.prior_cov_rvs(size)
        posterior_means = []

        for prior_cov in prior_covs:
            prior_mean = self.prior_mean_rvs(prior_cov)
            xi = self.cov @ np.linalg.inv(prior_cov + self.cov)
            delta = np.identity(self.mean.shape[0]) - xi
            expected_post_mean = prior_mean + delta @ (self.mean - prior_mean)
            dist = multivariate_normal(expected_post_mean, delta @ self.cov)
            posterior_means.append(dist.rvs())

        return np.array(posterior_means), sample_weight

    def _scaled_log_likelihood(self, prior_cov: np.ndarray) -> float:
        """Compute the (scaled) log likelihood of observing a prior covariance matrix
        given the sample mean and covariance matrix.

        This method is used to determine sample weights for Gibbs sampling.

        Args:
            prior_cov (np.ndarray): (n, n) prior covariance matrix.

        Raises:
            NotImplementedError: Classes which inherit the mixin should implement this
                method.

        Returns:
            float: (Scaled) log likelihood.
        """
        raise NotImplementedError()  # pragma: no cover


class LinearHierarchicalBayes(HierarchicalBayesBase):
    """Hierarchical linear Bayesian model.

    Inherits from :class:`HierarchicalBayesBase`.

    Assumes the prior mean vector is a linear combination of the feature matrix and
    that the prior covariance matrix is proportional to the identity matrix.

    Examples:

        .. code-block::

            >>> import numpy as np
            >>> from conditional_inference.bayes.hierarchical import LinearHierarchicalBayes
            >>> from scipy.stats import multivariate_normal, loguniform
            >>> n_policies = 5
            >>> prior_cov_params_distribution = loguniform(.1, 10)
            >>> prior_cov = prior_cov_params_distribution.rvs() * np.identity(n_policies)
            >>> prior_mean = np.zeros(n_policies)
            >>> true_mean = multivariate_normal.rvs(prior_mean, prior_cov)
            >>> sample_cov = np.identity(n_policies)
            >>> sample_mean = multivariate_normal.rvs(true_mean, sample_cov)
            >>> model = LinearHierarchicalBayes(sample_mean, sample_cov, prior_cov_params_distribution)
            >>> model.fit(cols="sorted").summary()
             Hierarchical Bayes estimates
            ==============================
                coef  pvalue [0.025 0.975]
            ------------------------------
            x0  1.545  0.041 -0.165  3.531
            x1  0.413  0.348 -1.478  2.317
            x2  0.041  0.479 -1.961  2.024
            x4 -1.200  0.900 -2.940  0.673
            x3 -3.772  1.000 -5.828 -1.620
            ===============
            Dep. Variable y
            ---------------
    """

    def estimate_prior_cov(self, prior_cov_params: float) -> np.ndarray:  # type: ignore
        """Estimate the prior covariance matrix.

        Args:
            prior_cov_params (float): Standard deviation of the prior distribution.

        Returns:
            np.ndarray: (n, n) prior covariance matrix. Assumed to be proportional to
                the identity matrix.
        """
        return prior_cov_params ** 2 * np.identity(self.mean.shape[0])

    def prior_mean_rvs(self, prior_cov: np.ndarray, size: int = 1) -> np.ndarray:
        """Sample means from distribution of prior means.

        Args:
            prior_cov (np.ndarray): (n, n) prior covariance matrix.
            size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: (size, n) array of prior mean samples.
        """
        X_T = self.X.T
        tau_inv = np.linalg.inv(prior_cov + self.cov)
        XT_tauinv_X_inv = np.linalg.inv(X_T @ tau_inv @ self.X)
        beta_bar = XT_tauinv_X_inv @ X_T @ tau_inv @ self.mean
        beta = multivariate_normal.rvs(beta_bar, XT_tauinv_X_inv, size=size)
        return (self.X @ beta.reshape(1, -1)).squeeze()

    def _scaled_log_likelihood(self, prior_cov: np.ndarray) -> float:
        # compute the scaled log likelihood; see HierarchicalBayesBase
        X_T = self.X.T
        tau = prior_cov + self.cov
        tau_inv = np.linalg.inv(tau)
        XT_tauinv_X_inv = np.linalg.inv(X_T @ tau_inv @ self.X)

        mean_bar = self.X @ XT_tauinv_X_inv @ self.X.T @ tau_inv @ self.mean
        error = self.mean - mean_bar
        return -0.5 * (
            np.log(np.linalg.det(tau))
            - np.log(np.linalg.det(XT_tauinv_X_inv))
            + error.T @ tau_inv @ error
        )


class HierarchicalBayesResults(BayesResults):
    """Results from hierarchical Bayesian analysis.

    Inherits from
    :class:`conditional_inference.bayes.base.BayesResults`.

    Args:
        model (HierarchicalBayesBase): Model on which the results are based.
        cols (ColumnsType): Columns of interest.
        posterior_mean_rvs (np.ndarray): (n_samples, n) array of samples from
            distribution of posterior means.
        sample_weight (np.ndarray, optional): (n_samples) array of sample weights.
            Defaults to None.
        title (str, optional): Results title. Defaults to
            "Hierarchical Bayes results".
    """

    def __init__(
        self,
        model: HierarchicalBayesBase,
        cols: Optional[ColumnsType],
        posterior_mean_rvs: np.ndarray,
        sample_weight: np.ndarray = None,
        title: str = "Hierarchical Bayes results",
    ):
        self.model = model
        self.indices = model.get_indices(cols)
        self.title = title

        self.posterior_mean_rvs = posterior_mean_rvs[:, self.indices]

        if sample_weight is None:
            sample_weight = np.ones(self.posterior_mean_rvs.shape[0])
        sample_weight = np.array(sample_weight)
        self.sample_weight = np.array(sample_weight) / sample_weight.sum()
        self.params = (self.posterior_mean_rvs.T * self.sample_weight).sum(axis=1)

        self.pvalues = np.apply_along_axis(
            weighted_cdf,
            0,
            self.posterior_mean_rvs,
            x=0,
            sample_weight=self.sample_weight,
        )
        self.rank_matrix = self._compute_rank_matrix()

    def conf_int(self, alpha: float = 0.05, cols: ColumnsType = None) -> np.ndarray:
        """Compute the 1-alpha confidence interval.

        Args:
            alpha (float, optional): The CI will cover the truth with probability
                1-alpha. Defaults to 0.05.
            cols (ColumnsType, optional): Names or indices of policies of interest.
                Defaults to None.

        Returns:
            np.ndarray: (n,2) array of confidence intervals.
        """
        indices = self._get_indices(cols)
        return np.array(
            [
                weighted_quantile(col, [alpha / 2, 1 - alpha / 2], self.sample_weight)
                for idx, col in enumerate(self.posterior_mean_rvs.T)
                if idx in indices
            ]
        )

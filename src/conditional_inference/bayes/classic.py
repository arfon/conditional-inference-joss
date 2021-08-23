"""Classical Bayesian analysis
"""
from __future__ import annotations

from typing import Any, Union

import numpy as np

from ..base import ColumnsType, Numeric1DArray
from .base import BayesModelBase, BayesResults


class ClassicBayesBase(BayesModelBase):
    """Mixin for classical Bayesian analysis.

    Inherits from :class:`conditional_inference.bayes.base.BayesModelBase`.

    Assumes a know prior covariance.

    Args:
        mean (Numeric1DArray): (n,) array of conventionally-estimated means.
        cov (np.ndarray): (n, n) covariance matrix.
        prior_cov (Union[float, np.ndarray]): (n, n) prior covariance matrix. If
            ``float``, the prior covariance is assumed to be proportional to the
            identity matrix.
        X (np.ndarray, optional): (n, p) feature matrix. If ``None``, a constant
            regressor will be used. Defaults to None.
        *args (Any): Passed to ``BayesModelBase``.
        **kwargs (Any): Passed to ``BayesModelBase``.

    Attributes:
        prior_cov (np.ndarray): (n, n) prior covariance matrix.

    Notes:
        This class also contains the attributes of ``BayesModelBase``.
    """

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        prior_cov: Union[float, np.ndarray],
        *args: Any,
        X: np.ndarray = None,
        **kwargs: Any
    ):
        super().__init__(mean, cov, *args, X=X, **kwargs)
        if np.isscalar(prior_cov):
            self.prior_cov = np.diag(np.full(len(mean), prior_cov))
        else:
            self.prior_cov = prior_cov

    def estimate_prior_mean(self, prior_mean_params=None) -> np.ndarray:
        """Estimate the prior mean vector.

        Args:
            prior_mean_params (Any, optional): Parameters which determine the prior
                mean. Defaults to None.

        Raises:
            NotImplementedError: Classes which inherit the mixin should implement this
                method.

        Returns:
            np.ndarray: (n,) array of prior means.
        """
        raise NotImplementedError()  # pragma: no cover

    def _estimate_prior_mean_params(self):
        """Estimate prior mean parameters.

        Raises:
            NotImplementedError: Classes which inherit the mixin should implement this
                method.
        """
        raise NotImplementedError()  # pragma: no cover

    def estimate_posterior_mean(self, prior_mean: np.ndarray = None) -> np.ndarray:
        """Estimate the posterior mean vector.

        Args:
            prior_mean (np.ndarray, optional): (n,) array of prior means. Defaults to
                None.

        Returns:
            np.ndarray: (n,) array of posterior means.
        """
        if prior_mean is None:
            prior_mean = self.estimate_prior_mean()
        xi = self._compute_xi(self.prior_cov)
        return prior_mean + (np.identity(self.mean.shape[0]) - xi) @ (
            self.mean - prior_mean
        )

    def estimate_posterior_cov(self) -> np.ndarray:
        """Estimate posterior covariance matrix.

        Returns:
            np.ndarray: (n, n) posterior covariance matrix.
        """
        xi = self._compute_xi(self.prior_cov)
        return (np.identity(self.mean.shape[0]) - xi) @ self.cov

    def fit(
        self,
        cols: ColumnsType = None,
        title: str = "Classical Bayes estimates",
        **kwargs: Any
    ) -> BayesResults:
        """Fit the model

        Args:
            cols (ColumnsType, optional): Columns of interest. Defaults to None.
            title (str, optional): Results title. Defaults to
                "Classical Bayes estimates".
            **kwargs (Any): Passed to ``BayesResults``.

        Returns:
            BayesResults: Results.
        """
        return BayesResults(
            self,
            cols,
            params=self.estimate_posterior_mean(),
            cov_params=self.estimate_posterior_cov(),
            title=title,
            **kwargs
        )


class LinearClassicBayes(ClassicBayesBase):
    """Classic linear Bayesian model. Inherits from :class:`ClassicBayesBase`.

    Assumes the prior mean vector is a linear combination of the feature matrix.

    Examples:

        .. code-block::

            >>> import numpy as np
            >>> from conditional_inference.bayes.classic import LinearClassicBayes
            >>> from scipy.stats import multivariate_normal
            >>> n_policies = 5
            >>> prior_cov = np.identity(n_policies)
            >>> prior_mean = np.zeros(n_policies)
            >>> true_mean = multivariate_normal.rvs(prior_mean, prior_cov)
            >>> sample_cov = np.identity(n_policies)
            >>> sample_mean = multivariate_normal.rvs(true_mean, sample_cov)
            >>> model = LinearClassicBayes(sample_mean, sample_cov, prior_cov=prior_cov)
            >>> model.fit(cols="sorted").summary()
              Classical Bayes estimates
            ==============================
                coef  pvalue [0.025 0.975]
            ------------------------------
            x2 -0.813  0.853 -2.331  0.705
            x0 -1.053  0.913 -2.571  0.465
            x1 -1.664  0.984 -3.182 -0.146
            x3 -1.782  0.989 -3.300 -0.263
            x4 -1.830  0.991 -3.348 -0.312
            ===============
            Dep. Variable y
            ---------------
    """

    def estimate_prior_mean(self, prior_mean_params: np.ndarray = None) -> np.ndarray:
        """Estimate the prior mean vector.

        Args:
            prior_mean_params (np.ndarray, optional): (p,) array of prior mean
                parameters. Defaults to None.

        Returns:
            np.ndarray: (n,) array of prior means.
        """
        if prior_mean_params is None:
            prior_mean_params = self._estimate_prior_mean_params()
        return self.X @ prior_mean_params

    def _estimate_prior_mean_params(self) -> np.ndarray:
        """Estimate the prior mean parameters.

        Returns:
            np.ndarray: (p,) array of prior mean parameters.
        """
        # tau is the covariance of marginal joint distribution of mean
        X_T = self.X.T
        tau_inv = np.linalg.inv(self.prior_cov + self.cov)
        if self._prior_is_infinite():
            return np.linalg.inv(X_T @ self.X) @ X_T @ self.mean
        return np.linalg.inv(X_T @ tau_inv @ self.X) @ X_T @ tau_inv @ self.mean

    def estimate_posterior_cov(self) -> np.ndarray:
        """Estimate the posterior covariance matrix

        Returns:
            np.ndarray: (n, n) posterior covariance matrix.
        """
        post_mean_uncertainty = super().estimate_posterior_cov()
        # increase posterior covariance to account for uncertainty in prior mean
        # parameters
        if self._prior_is_infinite():
            # prior mean uncertainty converges to 0
            return post_mean_uncertainty
        xi = self._compute_xi(self.prior_cov)
        X_T = self.X.T
        tau_inv = np.linalg.inv(self.prior_cov + self.cov)
        prior_mean_uncertainty = (
            xi @ self.X @ np.linalg.inv(X_T @ tau_inv @ self.X) @ X_T @ xi
        )
        return post_mean_uncertainty + prior_mean_uncertainty

    def _prior_is_infinite(self) -> bool:
        """Indicates that the prior covariance is ``np.inf * np.identity(n)``.

        Returns:
            bool: Indicator
        """
        return (self.prior_cov == np.diag(np.full(self.mean.shape[0], np.inf))).all()

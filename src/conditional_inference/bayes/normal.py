"""Empirical Bayes with a normal prior.

References:

    .. code-block::

        @inproceedings{stein1956inadmissibility,
            title={Inadmissibility of the usual estimator for the mean of a multivariate normal distribution},
            author={Stein, Charles and others},
            booktitle={Proceedings of the Third Berkeley symposium on mathematical statistics and probability},
            volume={1},
            number={1},
            pages={197--206},
            year={1956}
        }

        @incollection{james1992estimation,
            title={Estimation with quadratic loss},
            author={James, William and Stein, Charles},
            booktitle={Breakthroughs in statistics},
            pages={443--460},
            year={1992},
            publisher={Springer}
        }

        @article{bock1975minimax,
            title={Minimax estimators of the mean of a multivariate normal distribution},
            author={Bock, Mary Ellen},
            journal={The Annals of Statistics},
            pages={209--218},
            year={1975},
            publisher={JSTOR}
        }

        @inproceedings{dimmery2019shrinkage,
            title={Shrinkage estimators in online experiments},
            author={Dimmery, Drew and Bakshy, Eytan and Sekhon, Jasjeet},
            booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
            pages={2914--2922},
            year={2019}
        }

Notes:

    The James-Stein method of fitting the normal prior relies on my own fully Bayesian
    derivation that extends Dimmery et al. (2019)'s derivation by 1) accounting for correlated
    errors and 2) allowing the prior mean vector to depend on a feature matrix ``X``.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Callable, Union

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import multivariate_normal, norm, rv_continuous

from conditional_inference.bayes.base import BayesBase


class Normal(BayesBase):
    """Bayesian model with a normal prior.

    Args:
        fit_method (Union[str, Callable[[], None]], optional): Specifies how to fit the
            prior ("mle", "bock", or "james_stein"). You can also use a custom function
            that sets the ``prior_mean``, ``prior_cov``, ``posterior_mean`` and
            ``posterior_cov`` attributes. Defaults to "mle".
        prior_mean (Union[float, np.ndarray], optional): (# params,) prior mean vector.
            Defaults to None.
        prior_cov (Union[float, np.ndarray], optional): (# params, # params) prior
            covariance matrix. Defaults to None.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.bayes import Normal

            model = Normal(np.arange(10), np.identity(10))
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                       Bayesian estimates
            =======================================
                coef pvalue (1-sided) [0.025 0.975]
            ---------------------------------------
            x0 0.545            0.282 -1.305  2.395
            x1 1.424            0.066 -0.426  3.274
            x2 2.303            0.007  0.453  4.153
            x3 3.182            0.000  1.332  5.032
            x4 4.061            0.000  2.211  5.911
            x5 4.939            0.000  3.089  6.789
            x6 5.818            0.000  3.968  7.668
            x7 6.697            0.000  4.847  8.547
            x8 7.576            0.000  5.726  9.426
            x9 8.455            0.000  6.605 10.305
            ===============
            Dep. Variable y
            ---------------
    """

    def __init__(
        self,
        *args: Any,
        fit_method: Union[str, Callable[[], None]] = "mle",
        prior_mean: Union[float, np.ndarray] = None,
        prior_cov: Union[float, np.ndarray] = None,
        **kwargs: Any,
    ):

        super().__init__(*args, **kwargs)
        self.prior_mean, self.prior_cov = prior_mean, prior_cov
        if np.isscalar(prior_mean):
            self.prior_mean = np.full(self.n_params, prior_mean)
        if np.isscalar(prior_cov):
            self.prior_cov = prior_cov * np.identity(self.n_params)

        self.posterior_mean, self.posterior_cov = None, None
        if callable(fit_method):
            fit_method()
            self._set_posterior_estimates()
        else:
            fit_methods = {
                "mle": self._fit_mle,
                "james_stein": self._fit_james_stein,
                "bock": self._fit_bock,
            }
            if fit_method not in fit_methods:
                raise ValueError(
                    f"`fit_method` must be one of {fit_methods.keys()}, got {fit_method}."
                )
            fit_methods[fit_method]()

    def _fit_mle(self, max_iter: int = 100, rtol: float = 0.99) -> None:
        """Fit the model using maximum likelihood estimation.

        Args:
            max_iter (int, optional): Maximum number of EM iterations. Defaults to 100.
            rtol (float, optional): Stopping criterion for EM. Defaults to .99.
        """

        def neg_log_likelihood(prior_std):
            # negative log likelihood as a function of the prior standard deviation
            marginal_cov = prior_std ** 2 * np.identity(self.n_params) + self.cov
            # note: the prior mean is also the marginal mean
            return -multivariate_normal.logpdf(self.mean, prior_mean, marginal_cov)

        # use EM to iteratively update the prior mean and covariance
        prior_cov = (
            np.zeros(self.cov.shape) if self.prior_cov is None else self.prior_cov
        )
        current_log_likelihood, prev_log_likelihood = None, -np.inf
        for _ in range(max_iter):
            # update prior mean
            if self.prior_mean is not None:
                prior_mean = self.prior_mean
            else:
                marginal_cov_inv = np.linalg.inv(self.cov + prior_cov)
                prior_mean = (
                    self.X
                    @ np.linalg.inv(self.X.T @ marginal_cov_inv @ self.X)
                    @ self.X.T
                    @ marginal_cov_inv
                    @ self.mean
                )

            # update prior cov
            if self.prior_cov is not None:
                prior_cov = self.prior_cov
                break  # prior_mean is computed analytically, so no need to iterate futher
            else:
                result = minimize_scalar(
                    neg_log_likelihood, bounds=(0, self.mean.std()), method="bounded"
                )
                prior_cov = result.x ** 2 * np.identity(self.n_params)
                current_log_likelihood = -result.fun

            if current_log_likelihood / prev_log_likelihood > rtol:
                break
            prev_log_likelihood = current_log_likelihood

        # set the posterior mean and covariance estimates
        # and adjust the prior and posterior covariances to account for uncertainty in the MLE estimate of the prior mean
        prior_uncertainty = post_uncertainty = 0
        if self.prior_mean is None:
            marginal_cov_inv = np.linalg.inv(prior_cov + self.cov)
            prior_uncertainty = (
                self.X @ np.linalg.inv(self.X.T @ marginal_cov_inv @ self.X) @ self.X.T
            )
            xi = self.cov @ marginal_cov_inv
            post_uncertainty = xi @ prior_uncertainty @ xi

        self.prior_mean, self.prior_cov = prior_mean, prior_cov
        self._set_posterior_estimates()  # note: set the posterior estimates *before* adjusting the prior covariance
        self.prior_cov += prior_uncertainty
        self.posterior_cov += post_uncertainty

    def _fit_bock(self, max_iter: int = 100, rtol: float = 0.99) -> None:
        """Fit the model using Bock (1975)'s multivariate Stein-type estimator.

        Args:
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            rtol (float, optional): Stopping criteria. Defaults to .99.

        Raises:
            RuntimeError: The shrinkage factor must be positve.
        """
        cov_inv = np.linalg.inv(self.cov)
        prior_mean_df = self.X.shape[1] if self.prior_mean is None else 0
        effective_dimension = np.trace(self.cov) / np.linalg.eig(self.cov)[0].max()
        if effective_dimension - prior_mean_df - 2 < 0:
            raise RuntimeError(
                "Failed to fit the Bock (1975) estimator because the effective dimension"
                " of the covariance matrix is too small. Try another fit method like"
                " 'mle'."
            )

        xi = (
            np.identity(self.n_params)
            if self.prior_cov is None
            else self.cov @ np.linalg.inv(self.cov + self.prior_cov)
        )
        current_log_likelihood, prev_log_likelihood = None, -np.inf
        for _ in range(max_iter):
            if self.prior_mean is None:
                # update prior mean
                marginal_cov_inv = cov_inv @ xi
                prior_mean = (
                    self.X
                    @ np.linalg.inv(self.X.T @ marginal_cov_inv @ self.X)
                    @ self.X.T
                    @ marginal_cov_inv
                    @ self.mean
                )
            else:
                prior_mean = self.prior_mean

            if self.prior_cov is None:
                # update prior covariance
                error = self.mean - prior_mean
                param = min(
                    (effective_dimension - prior_mean_df - 2)
                    / (error.T @ cov_inv @ error),
                    1,
                )
                xi = param * np.identity(self.n_params)
            else:
                prior_cov = self.prior_cov
                # prior mean is computed analytically, so no need to iterate
                break

            # check for convergence
            marginal_cov = np.linalg.inv(xi) @ self.cov
            prior_cov = marginal_cov - self.cov
            current_log_likelihood = multivariate_normal.logpdf(
                self.mean, prior_mean, marginal_cov
            )
            if current_log_likelihood / prev_log_likelihood > rtol:
                break
            prev_log_likelihood = current_log_likelihood

        # set the posterior mean and covariance estimates
        # and adjust the prior and posterior covariances to account for uncertainty in the MLE estimate of the prior mean
        prior_uncertainty = post_uncertainty = 0
        if self.prior_mean is None:
            marginal_cov_inv = np.linalg.inv(prior_cov + self.cov)
            prior_uncertainty = (
                self.X @ np.linalg.inv(self.X.T @ marginal_cov_inv @ self.X) @ self.X.T
            )
            xi = self.cov @ marginal_cov_inv
            post_uncertainty = xi @ prior_uncertainty @ xi

        self.prior_mean, self.prior_cov = prior_mean, prior_cov
        self._set_posterior_estimates()  # note: set the posterior estimates *before* adjusting the prior covariance
        self.prior_cov += prior_uncertainty
        self.posterior_cov += post_uncertainty

    def _set_posterior_estimates(self):
        """Sets the posterior mean and covariance using plugin estimates from the prior
        mean and covariance if the posterior parameters haven't already been set.
        """
        xi = self.cov @ np.linalg.inv(self.prior_cov + self.cov)
        if self.posterior_mean is None:
            self.posterior_mean = self.prior_mean + (
                np.identity(self.n_params) - xi
            ) @ (self.mean - self.prior_mean)

        if self.posterior_cov is None:
            self.posterior_cov = (np.identity(self.n_params) - xi) @ self.cov

    def _fit_james_stein(self, max_iter: int = 100, tol: float = 1e-6) -> None:
        """Fit the model using James-Stein estimates.

        Args:
            max_iter (int, optional): Maximum number of iterations to find the
                positive-part prior covariance.. Defaults to 100.
            tol (float, optional): Stopping criteria for finding the positive-part prior
                covariance. Defaults to 1e-6.
        """
        if self.prior_mean is None:
            prior_mean = (
                self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.mean
            )
            prior_mean_df = self.X.shape[1]
        else:
            prior_mean = self.prior_mean
            prior_mean_df = 0

        s_squared = ((self.mean - prior_mean) ** 2).sum()
        try:
            np.linalg.cholesky(
                s_squared
                / (self.n_params - prior_mean_df - 2)
                * np.identity(self.n_params)
                - self.cov
            )
        except np.linalg.LinAlgError:
            # find minimum s_squared such that the (unadjusted) prior covariance is positive semidefinite
            warnings.warn(
                "The James-Stein prior covariance estimate is not positive semidefinite."
                " Using the positive-part James-Stein covariance estimate instead."
                " This may result in too little shrinkage."
            )
            bounds = [np.sqrt(s_squared), np.inf]
            for _ in range(max_iter):
                s_squared = (
                    2 * bounds[0] if bounds[1] == np.inf else sum(bounds) / 2
                ) ** 2
                try:
                    np.linalg.cholesky(
                        s_squared
                        / (self.n_params - prior_mean_df - 2)
                        * np.identity(self.n_params)
                        - self.cov
                    )
                    bounds[1] = np.sqrt(s_squared)
                except:
                    bounds[0] = np.sqrt(s_squared)
                if bounds[1] - bounds[0] < tol:
                    break
            s_squared = bounds[1] ** 2

        # compute the prior covariance
        param = s_squared / (self.n_params - prior_mean_df - 4)
        self.prior_cov = (
            param * np.identity(self.n_params)
            - self.cov
            + param * self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T
        )

        # compute the posterior mean
        xi = self.cov * (self.n_params - prior_mean_df - 2) / s_squared
        self.posterior_mean = prior_mean + (np.identity(self.n_params) - xi) @ (
            self.mean - prior_mean
        )

        # compute the posterior covariance
        plugin_posterior_cov = (np.identity(self.n_params) - xi) @ self.cov
        if self.prior_mean is None:
            prior_mean_uncertainty = (
                self.cov @ self.X @ np.linalg.inv(self.X.T @ self.X) @ self.X.T @ xi
            )
        else:
            prior_mean_uncertainty = 0
        prior_cov_uncertainty = (
            2
            / (self.n_params - self.X.shape[1] - 2)
            * xi
            @ (self.mean - prior_mean).reshape(-1, 1)
            @ (self.mean - prior_mean).reshape(1, -1)
            @ xi
        )
        self.posterior_cov = (
            plugin_posterior_cov + prior_mean_uncertainty + prior_cov_uncertainty
        )

        self.prior_mean = prior_mean

    def _get_marginal_prior(self, index: int) -> rv_continuous:
        return norm(self.prior_mean[index], np.sqrt(self.prior_cov[index, index]))

    def _get_marginal_distribution(self, index: int) -> rv_continuous:
        return norm(
            self.posterior_mean[index], np.sqrt(self.posterior_cov[index, index])
        )

    def _get_joint_prior(self, indices: np.ndarray):
        return multivariate_normal(
            self.prior_mean[indices],
            self.prior_cov[indices][:, indices],
            allow_singular=True,
        )

    def _get_joint_distribution(self, indices: np.ndarray):
        return multivariate_normal(
            self.posterior_mean[indices],
            self.posterior_cov[indices][:, indices],
            allow_singular=True,
        )

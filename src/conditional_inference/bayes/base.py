"""Base classes for Bayesian analysis
"""
from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, multivariate_normal, wasserstein_distance

from ..base import ModelBase, Numeric1DArray, ResultsBase, ColumnsType
from ..utils import expected_wasserstein_distance, weighted_quantile


class BayesModelBase(ModelBase):
    """Mixin for Bayesian models.

    Inherits from :class:`conditional_inference.base.ModelBase`.

    Args:
        mean (Numeric1DArray): (n,) array of conventionally-estimated means.
        cov (np.ndarray): (n, n) covariance matrix.
        X (np.ndarray, optional): (n, p) feature matrix. If ``None``, a constant
            regressor will be used. Defaults to None.
        *args (Any): Passed to ``ModelBase``.
        **kwargs (Any): Passed to ``ModelBase``.

    Attributes:
        X (np.ndarray): (n, p) feature matrix.
    """

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        *args: Any,
        X: np.ndarray = None,
        **kwargs: Any,
    ):
        super().__init__(mean, cov, *args, **kwargs)
        if X is None:
            self.X = np.ones((len(mean), 1))
        elif hasattr(X, "values"):
            # assume X.values is array-like
            self.X = X.values  # type: ignore
        else:
            self.X = X

    def _compute_xi(self, prior_cov: np.ndarray) -> np.ndarray:
        """Compute xi; see paper for mathematical detail.

        Args:
            prior_cov (np.ndarray): (n, n) prior covariance matrix.

        Returns:
            np.ndarray: (n, n) weight matrix.
        """
        return self.cov @ np.linalg.inv(prior_cov + self.cov)


class BayesResults(ResultsBase):
    """Results of Bayesian analysis.

    Inherits from :class:`conditional_inference.base.ResultsBase`.

    Args:
        model (BayesModelBase): Model on which results are based.
        cols (ColumnsType): Columns of interest.
        params (np.ndarray): (n,) array of point estimates, usually the average
            posterior mean.
        cov_params (np.ndarray): (n, n) posterior covariance matrix.
        n_samples (int, optional): Number of samples to draw for approximations, such
            as likelihood calculations. Defaults to 1000.
        title (str, optional): Results title. Defaults to "Bayesian estimates".

    Attributes:
        params (np.ndarray): (n,) array of point estimates, usually the average
            posterior mean.
        cov_params (np.ndarray): (n, n) posterior covariance matrix.
        distributions (List[scipy.stats.norm]): Marginal posterior distributions.
        multivariate_distribution (scipy.stats.multivariate_normal): Joint posterior
            distribution.
        pvalues (np.ndarray): (n,) array of probabilities that the true mean is less
            than 0.
        posterior_mean_rvs (np.ndarray): (n_samples, n) matrix of draws from the
            posterior.
        rank_matrix (pd.DataFrame): (n, n) dataframe of probabilities that column i has
            rank j.
    """

    def __init__(
        self,
        model: BayesModelBase,
        cols: Optional[ColumnsType],
        params: np.ndarray,
        cov_params: np.ndarray,
        n_samples: int = 1000,
        seed: int = 0,
        title: str = "Bayesian estimates",
    ):
        super().__init__(model, cols, title)

        self.params = params[self.indices]
        self.cov_params = cov_params[self.indices][:, self.indices]
        self.distributions = [
            norm(params[k], np.sqrt(cov_params[k, k])) for k in self.indices
        ]
        self.pvalues = np.array([dist.cdf(0) for dist in self.distributions])
        self.sample_weight = np.full(n_samples, 1 / n_samples)
        self.seed = seed

        try:
            self.multivariate_distribution = multivariate_normal(
                self.params, self.cov_params
            )
            self.posterior_mean_rvs = self.multivariate_distribution.rvs(
                n_samples, random_state=seed
            )
            self.rank_matrix = self._compute_rank_matrix()
        except np.linalg.LinAlgError:
            # the policy effects are perfectly correlated
            # this occurs when the prior covariance == 0
            warnings.warn("Posterior covariance matrix is singular")
            self.multivariate_distribution = None
            err = norm.rvs(
                0, np.sqrt(self.cov_params[0, 0]), size=n_samples, random_state=seed
            )
            self.posterior_mean_rvs = self.params + np.repeat(
                err.reshape(-1, 1), self.params.shape[0], axis=1
            )
            self.rank_matrix = self._compute_rank_matrix(singular=True)

    @property
    def reconstructed_mean_rvs(  # pylint: disable=missing-function-docstring
        self,
    ) -> np.ndarray:
        # reconstruct means and cache the value if they have not been created already
        def reconstruct_means(mean):
            return multivariate_normal.rvs(mean, self.model.cov)

        if not hasattr(self, "_reconstructed_mean_rvs"):
            self.reconstructed_mean_rvs = np.apply_along_axis(
                reconstruct_means, 1, self.posterior_mean_rvs
            )

        return self._reconstructed_mean_rvs

    @reconstructed_mean_rvs.setter
    def reconstructed_mean_rvs(self, value: np.ndarray) -> None:
        self._reconstructed_mean_rvs = value

    def expected_wasserstein_distance(
        self, mean: Numeric1DArray = None, cov: np.ndarray = None, **kwargs: Any
    ) -> float:
        """Compute the Wasserstein distance metric.

        This method estimates the Wasserstein distance between the observed
        distribution (a joint normal characterized by ``mean`` and ``cov``) and the
        distribution you would expect to observe according to this model.

        Args:
            mean (Numeric1DArray, optional): (n,) array of sample means. Defaults to
                None.
            cov (np.ndarray, optional): (n, n) covaraince matrix for sample means.
                Defaults to None.
            **kwargs (Any): Keyword arguments for ``scipy.stats.wasserstein_distance``.

        Returns:
            float: Expected Wasserstein distance.

        Note:
            ``mean`` and ``cov`` are taken to be the mean and covariance used to fit the
            model by default, giving you the in-sample Wasserstein distance.
        """

        def compute_distance(reconstructed_mean):
            return wasserstein_distance(reconstructed_mean, self.model.mean, **kwargs)

        if mean is None and cov is None:
            distances = np.apply_along_axis(
                compute_distance, 1, self.reconstructed_mean_rvs
            )
            return (self.sample_weight * distances).sum()

        mean = self.model.mean[self.indices] if mean is None else mean
        cov = self.model.cov[self.indices][:, self.indices] if cov is None else cov
        return expected_wasserstein_distance(
            mean, cov, self.posterior_mean_rvs, self.sample_weight, **kwargs
        )

    def likelihood(self, mean: Numeric1DArray = None, cov: np.ndarray = None) -> float:
        """Compute the likelihood of observing the sample means.

        Args:
            mean (Numeric1DArray, optional): (n,) array of sample means. Defaults to
                None.
            cov (np.ndarray, optional): (n, n) covariance matrix for sample means.
                Defaults to None.

        Returns:
            float: Likelihood.

        Note:
            ``mean`` and ``cov`` are taken to be the mean and covariance used to fit the
            model by default, giving you the in-sample likelihood.
        """
        mean = self.model.mean[self.indices] if mean is None else mean
        cov = self.model.cov[self.indices][:, self.indices] if cov is None else cov
        likelihood = np.apply_along_axis(
            lambda params: multivariate_normal(params, cov).pdf(mean),
            1,
            self.posterior_mean_rvs,
        )
        return (self.sample_weight * likelihood).sum()

    def rank_matrix_plot(self, *args: Any, title: str = None, **kwargs: Any):
        """Plot a heatmap of the rank matrix.

        Args:
            title (str, optional): Plot title. Defaults to None.
            *args (Any): Passed to ``sns.heatmap``.
            **kwargs (Any): Passed to ``sns.heatmap``.

        Returns:
            AxesSubplot: Heatmap.
        """
        ax = sns.heatmap(
            self.rank_matrix, center=1 / self.params.shape[0], *args, **kwargs
        )
        ax.set_title(title or self.title)
        return ax

    def reconstruction_histogram(
        self,
        yname: str = None,
        title: str = None,
        ax=None,
    ):
        """Create a histogram of the reconstructed means.

        Plots the distribution of sample means you would expect to see if this model
        were correct.

        Args:
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            title (str, optional): Plot title. Defaults to None.
            ax: (AxesSubplot, optional): Axis to write on.

        Returns:
            plt.axes._subplots.AxesSubplot: Plot.
        """
        params = np.sort(self.reconstructed_mean_rvs).mean(axis=0)

        if ax is None:
            _, ax = plt.subplots()
        sns.histplot(
            x=list(self.model.mean) + list(params),
            hue=len(self.model.mean) * ["Observed"] + len(params) * ["Reconstructed"],
            stat="probability",
            kde=True,
            ax=ax,
        )
        ax.set_title(title or f"{self.title} reconstruction plot")
        ax.set_xlabel(yname or self.model.endog_names)

        return ax

    def reconstruction_point_plot(
        self,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
        alpha: float = 0.05,
        ax=None,
    ):
        """Create  point plot of the reconstructed sample means.

        Plots the distribution of sample means you would expect to see if this model
        were correct.

        Args:
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): Names of x-ticks. Defaults to None.
            title (str, optional): Plot title. Defaults to None.
            alpha: (float, optional): Plot the 1-alpha CI. Defaults to 0.05.
            ax: (AxesSubplot, optional): Axis to write on.

        Returns:
            plt.axes._subplots.AxesSubplot: Plot.
        """
        reconstructed_means = -np.sort(-self.reconstructed_mean_rvs)
        params = reconstructed_means.mean(axis=0)

        weighted_quantile_func = partial(
            weighted_quantile,
            quantiles=[alpha / 2, 1 - alpha / 2],
            sample_weight=self.sample_weight,
        )
        conf_int = np.apply_along_axis(weighted_quantile_func, 0, reconstructed_means).T

        xname = xname or np.arange(len(self.indices))
        yticks = np.arange(len(xname), 0, -1)
        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(
            x=params,
            y=yticks,
            xerr=[params - conf_int[:, 0], conf_int[:, 1] - params],
            fmt="o",
        )
        ax.set_title(title or f"{self.title} reconstruction plot")
        ax.set_xlabel(yname or self.model.endog_names)
        ax.set_ylabel("rank")
        ax.set_yticks(yticks)
        ax.set_yticklabels(xname)

        ax.errorbar(x=-np.sort(-self.model.mean), y=yticks, fmt="x")

        return ax

    def _compute_rank_matrix(self, singular: bool = False) -> pd.DataFrame:
        """Compute the rank matrix

        Args:
            singular (bool, optional): Indicates the posterior covariance matrix is
                singular. Defaults to False.

        Returns:
            pd.DataFrame: Rank matrix.
        """
        if len(self.posterior_mean_rvs.shape) == 1:
            # only estimating one parameter
            rank_matrix = [1]
        elif not singular:
            # assumes no ties in rank order
            argsort = np.argsort(-self.posterior_mean_rvs, axis=1)
            rank_matrix = np.array(
                [
                    ((argsort == k).T * self.sample_weight).sum(axis=1)
                    for k in range(self.posterior_mean_rvs.shape[1])
                ]
            ).T
        else:
            # handles ties when posterior covariance matrix is singular
            rank_matrix = np.zeros((self.params.shape[0], self.params.shape[0]))
            params = self.params.copy()
            curr_rank = 0
            while params.shape[0] > 0:
                idx = np.where(self.params == params.max())[0]
                rank = (curr_rank + np.arange(idx.shape[0])).astype(int)
                for i in idx:
                    rank_matrix[rank, i] = 1 / idx.shape[0]
                curr_rank += idx.shape[0]
                params = params[params != params.max()]
        rank_df = pd.DataFrame(
            rank_matrix, columns=[self.model.exog_names[k] for k in self.indices]
        )
        rank_df.index.name = "Rank"
        return rank_df

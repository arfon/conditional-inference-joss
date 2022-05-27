"""Base classes for Bayesian analysis.
"""
from __future__ import annotations

import warnings
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal, norm, rv_continuous, wasserstein_distance

from ..base import ColumnType, ModelBase, Numeric1DArray, ResultsBase, ColumnsType
from ..stats import joint_distribution
from ..utils import weighted_quantile


class BayesResults(ResultsBase):
    """Results of Bayesian analysis.

    Inherits from :class:`conditional_inference.base.ResultsBase`.

    Args:
        *args (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
        n_samples (int): Number of samples used for approximations (ranking, likelihood
            and Wasserstein distance). Defaults to 10000.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ResultsBase`.

    Attributes:
        distributions (List[scipy.stats.norm]): Marginal posterior distributions.
        multivariate_distribution (scipy.stats.multivariate_normal): Joint posterior
            distribution.
        rank_matrix (pd.DataFrame): (n, n) dataframe of probabilities that column i has
            rank j.
    """

    _default_title = "Bayesian estimates"

    def __init__(self, *args: Any, n_samples: int = 10000, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # get the marginal (posterior) distributions, parameters, and pvalues
        self.marginal_distributions, params, pvalues = [], [], []
        for i in range(self.model.n_params):
            dist = self.model.get_marginal_distribution(i)
            self.marginal_distributions.append(dist)
            params.append(dist.mean())
            pvalues.append(dist.cdf(0))
        self.params = np.array(params).squeeze()
        self.pvalues = np.array(pvalues).squeeze()

        # estimate the parameter rankings by drawing from the posterior
        try:
            self.joint_distribution = self.model.get_joint_distribution()
            self._posterior_rvs = self.joint_distribution.rvs(size=n_samples)
            self._sample_weight = np.ones(n_samples)
        except NotImplementedError:
            warnings.warn(
                "Model does not provide a joint posterior distribution."
                " I'll assume the marginal posterior distributions are independent."
                " Rank estimates and likelihood and Wasserstein approximations may be"
                " unreliable."
            )
            self._posterior_rvs = joint_distribution(self.marginal_distributions).rvs(
                size=n_samples
            )
            self._sample_weight = np.ones(n_samples)
        self._sample_weight /= self._sample_weight.sum()
        argsort = np.argsort(-self._posterior_rvs, axis=1)
        rank_matrix = np.array(
            [
                ((argsort == k).T * self._sample_weight).sum(axis=1)
                for k in range(self.model.n_params)
            ]
        ).T
        self.rank_df = pd.DataFrame(
            rank_matrix,
            columns=self.model.exog_names,
            index=np.arange(1, self.model.n_params + 1),
        )
        self.rank_df.index.name = "Rank"

        self._reconstructed_rvs = np.apply_along_axis(
            lambda mean: multivariate_normal.rvs(mean, self.model.cov),
            1,
            self._posterior_rvs,
        )

    def expected_wasserstein_distance(
        self, mean: Numeric1DArray = None, cov: np.ndarray = None, **kwargs: Any
    ) -> float:
        """Compute the Wasserstein distance metric.

        This method estimates the Wasserstein distance between the observed
        distribution (a joint normal characterized by ``mean`` and ``cov``) and the
        distribution you would expect to observe according to this model.

        Args:
            mean (Numeric1DArray, optional): (# params,) array of sample conventionally
                estimated means. If None, use the model's estimated means. Defaults to
                None.
            cov (np.ndarray, optional): (# params, # params) covaraince matrix for
                conventionally estimated means. If None, use the model's estimated
                covariance matrix. Defaults to None.
            **kwargs (Any): Keyword arguments for ``scipy.stats.wasserstein_distance``.

        Returns:
            float: Expected Wasserstein distance.
        """
        if mean is None and cov is None:
            mean = self.params
            reconstructed_rvs = self._reconstructed_rvs
        else:
            if cov is None:
                cov = self.model.cov
            reconstructed_rvs = np.apply_along_axis(
                lambda mean: multivariate_normal.rvs(mean, cov), 1, self._posterior_rvs
            )

        distances = np.apply_along_axis(
            lambda rv: wasserstein_distance(rv, mean, **kwargs), 1, reconstructed_rvs
        )
        return (self._sample_weight * distances).sum()

    def likelihood(self, mean: Numeric1DArray = None, cov: np.ndarray = None) -> float:
        """
        Args:
            mean (Numeric1DArray, optional): (# params,) array of sample conventionally
                estimated means. If None, use the model's estimated means. Defaults to
                None.
            cov (np.ndarray, optional): (# params, # params) covaraince matrix for
                conventionally estimated means. If None, use the model's estimated
                covariance matrix. Defaults to None.

        Returns:
            float: Likelihood.
        """
        if mean is None:
            mean = self.model.mean
        if cov is None:
            cov = self.model.cov

        return (
            self._sample_weight
            * multivariate_normal.pdf(self._posterior_rvs, mean, cov)
        ).sum()

    def line_plot(
        self,
        column: ColumnType = None,
        alpha: float = 0.05,
        title: str = None,
        yname: str = None,
    ):
        """Create a line plot of the prior, conventional, and posterior estimates.

        Args:
            column (ColumnType, optional): Selected parameter. Defaults to None.
            alpha (float, optional): Sets the plot width. 0 is as wide as possible, 1 is
                as narrow as possible. Defaults to .05.
            title (str, optional): Plot title. Defaults to None.
            yname (str, optional): Name of the dependent variable. Defaults to None.

        Returns:
            AxesSubplot: Plot.
        """
        index = self.model.get_index(column)
        prior = self.model.get_marginal_prior(index)
        posterior = self.marginal_distributions[index]
        conventional = norm(
            self.model.mean[index], np.sqrt(self.model.cov[index, index])
        )
        xlim = np.array(
            [
                dist.ppf([alpha / 2, 1 - alpha / 2])
                for dist in (prior, conventional, posterior)
            ]
        ).T
        x = np.linspace(xlim[0].min(), xlim[1].max())
        palette = sns.color_palette()
        ax = sns.lineplot(x=x, y=prior.pdf(x), label="prior")
        ax.axvline(prior.mean(), linestyle="--", color=palette[0])
        sns.lineplot(x=x, y=conventional.pdf(x), label="conventional")
        ax.axvline(conventional.mean(), linestyle="--", color=palette[1])
        sns.lineplot(x=x, y=posterior.pdf(x), label="posterior")
        ax.axvline(posterior.mean(), linestyle="--", color=palette[2])
        ax.set_title(title or self.model.exog_names[index])
        ax.set_xlabel(yname or self.model.endog_names)
        return ax

    def rank_matrix_plot(self, title: str = None, **kwargs: Any):
        """Plot a heatmap of the rank matrix.

        Args:
            title (str, optional): Plot title. Defaults to None.
            **kwargs (Any): Passed to ``sns.heatmap``.

        Returns:
            AxesSubplot: Heatmap.
        """
        ax = sns.heatmap(self.rank_df, center=1 / self.model.n_params, **kwargs)
        ax.set_title(title or f"{self.title} rank matrix")
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
        reconstructed_means = -np.sort(-self._reconstructed_rvs)
        params = np.average(reconstructed_means, axis=0, weights=self._sample_weight)

        conf_int = np.apply_along_axis(
            weighted_quantile,
            0,
            reconstructed_means,
            quantiles=[alpha / 2, 1 - alpha / 2],
            sample_weight=self._sample_weight,
        ).T

        xname = xname or np.arange(self.model.n_params)
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

    def _make_summary_header(self, alpha: float) -> list[str]:
        return ["coef", "pvalue (1-sided)", f"[{alpha/2}", f"{1-alpha/2}]"]


class BayesBase(ModelBase):
    """Mixin for Bayesian models.

    Subclasses :class:`conditional_inference.base.ModelBase`.
    """

    _results_cls = BayesResults

    def get_marginal_prior(self, column: ColumnType) -> rv_continuous:
        """Get the marginal prior distribution of ``column``.

        Args:
            column (ColumnType): Name or index of the parameter of interest.

        Returns:
            rv_continuous: Prior distribution
        """
        return self._get_marginal_prior(self.get_index(column))

    def _get_marginal_prior(self, index: int) -> rv_continuous:
        """Private version of :meth:`self.get_marginal_prior`."""
        raise NotImplementedError()

    def get_marginal_distribution(self, column: ColumnType) -> rv_continuous:
        """Get the marginal posterior distribution of ``column``.

        Args:
            column (ColumnType): Name or index of the parameter of interest.

        Returns:
            rv_continuous: Posterior distribution.
        """
        return self._get_marginal_distribution(self.get_index(column))

    def _get_marginal_distribution(self, index: int) -> rv_continuous:
        """Private version of :meth:`self.get_marginal_distribution`."""
        raise NotImplementedError()

    def get_joint_prior(self, columns: ColumnsType = None):
        """Get the joint prior distribution.

        Args:
            columns (ColumnsType, optional): Selected columns. Defaults to None.

        Returns:
            rv_like: Joint distribution.
        """
        return self._get_joint_prior(self.get_indices(columns))

    def _get_joint_prior(self, indices: np.ndarray):
        """Private version of :meth:`self.get_joint_prior`."""
        return joint_distribution([self.get_marginal_prior(i) for i in indices])

    def get_joint_distribution(self, columns: ColumnsType = None):
        """Get the joint posterior distribution.

        Args:
            columns (ColumnsType, optional): Selected columns. Defaults to None.

        Returns:
            rv_like: Joint distribution.
        """
        return self._get_joint_distribution(self.get_indices(columns))

    def _get_joint_distribution(self, indices: np.ndarray):
        """Private version of :meth:`self.get_joint_distribution`."""
        raise NotImplementedError()

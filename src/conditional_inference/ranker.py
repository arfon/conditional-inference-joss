"""Corrections for multiple hypothesis tests and estimation of ranks.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from conditional_inference.base import ColumnsType, ModelBase, ResultsBase
from scipy.stats import multivariate_normal
import pandas as pd


class Ranker(ModelBase):
    """A model to conduct multiple hypothesis tests and estimate rankings.

    Subclasses :class:`conditional_inference.base.ModelBase`.
    """

    def fit(
        self, marginal=True, **kwargs: Any
    ) -> Union[MarginalResults, SimultaneousResults]:
        """Fits a model to estimate policy rankings.

        Args:
            marginal (bool, optional): Use marginal confidence intervals. If False, this
                will use simultaneous confidence intervals. Defaults to True.
            **kwargs (Any): Passed to :class:`MarginalResults` or :class:`SimultaneousResults`.

        Returns:
            Union[MarginalResults, SimultaneousResults]: Inference for rankings results.
        """
        return (
            MarginalResults(self, **kwargs)
            if marginal
            else SimultaneousResults(self, **kwargs)
        )


class RankerResultsBase(ResultsBase):
    """Base class for ranking results.

    Subclasses :class:`conditional_inference.base.ResultsBase`.
    """

    @staticmethod
    def _compute_delta_mean(mean: np.ndarray, i: int) -> np.ndarray:
        """Computes the difference in mean estimates.

        Args:
            mean (np.ndarray): (n,) array of estimated means.
            i (int): Rerence index.

        Returns:
            np.ndarray: (n - 1,) array of estimated differences between the
                reference mean and all other means.
        """
        return np.delete(mean[i] - mean, i)

    @staticmethod
    def _compute_delta_cov(cov: np.ndarray, i: int, j: int) -> np.ndarray:
        """Compute the covariance matrix for estimated differences between means.

        That is, given a vector of estimated means, this computes
        Cov(mean[i] - mean, mean[j] - mean).

        Args:
            cov (np.ndarray): (n, n) covariance matrix for estimated means.
            i (int): Reference index.
            j (int): Reference index.

        Returns:
            np.ndarray: (n - 1, n -1) covariance matrix for
                differences between estimated means.
        """
        repeat_i = np.repeat(np.atleast_2d(cov[i]), cov.shape[0], axis=0)
        repeat_j = np.repeat(np.atleast_2d(cov[j]), cov.shape[0], axis=0).T
        delta_cov = cov[i, j] + cov - repeat_i - repeat_j
        return np.delete(np.delete(delta_cov, i, axis=0), j, axis=1)

    @staticmethod
    def _stepdown(
        mean: np.ndarray, cov: np.ndarray, rvs: np.ndarray, alpha: float
    ) -> np.ndarray:
        """Use a stepdown procedure to reject null hypotheses.

        Args:
            mean (np.ndarray): (n,) array of estimated means.
            cov (np.ndarray): (n, n) covariance matrix.
            rvs (np.ndarray): (# samples, n) random values drawn from the joint distribution.
            alpha (float): Significance level.

        Returns:
            np.ndarray: (n,) boolean array indicating that hypothesis i was rejected.
        """
        rejected, newly_rejected = np.full(len(mean), False), None
        std_deviation = np.sqrt(cov.diagonal())
        while newly_rejected is None or (newly_rejected.any() and (~rejected).any()):
            projection_quantile = np.quantile(rvs[:, ~rejected].max(axis=1), 1 - alpha)
            newly_rejected = (
                mean - projection_quantile * std_deviation > 0
            ) & ~rejected
            rejected = newly_rejected | rejected

        return rejected

    def _select_hypothesis_matrix_columns(self, matrix, cols):
        indices = self.indices if cols is None else self.model.get_indices(cols)
        matrix = np.array(matrix).T[indices][:, indices]
        names = np.array(self.model.exog_names)[indices]
        return pd.DataFrame(matrix, columns=names, index=names)

    def compute_hypothesis_matrix(self, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError()

    def hypothesis_heatmap(self, *args, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()
        matrix = self.compute_hypothesis_matrix(*args, **kwargs)
        mask = np.zeros_like(matrix)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(
            self.compute_hypothesis_matrix(*args, **kwargs),
            cbar=False,
            ax=ax,
            yticklabels=self.model.exog_names,
            xticklabels=self.model.exog_names,
            # mask=mask,
            square=True,
        )
        ax.set_title(self.title)
        plt.yticks(rotation=0)
        return ax

    def conf_int(
        self, alpha: float = 0.05, cols: ColumnsType = None, **kwargs: Any
    ) -> np.ndarray:
        """Compute confidence intervals for the policy ranking.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            cols (ColumnsType, optional): Columns for which to compute the ranking.
                Defaults to None.
            **kwargs (Any): Passed to the ``compute_hypothesis_matrix`` method.

        Returns:
            np.ndarray: (# policies, 2) array of upper and lower bounds of the CI for
                each policy.
        """
        hypothesis_matrix = self.compute_hypothesis_matrix(
            alpha, cols=np.arange(len(self.model.mean)), **kwargs
        )
        conf_int = np.array(
            [
                hypothesis_matrix.sum(1),
                len(self.model.mean) - hypothesis_matrix.sum(0) - 1,
            ]
        ).T
        return conf_int[self.indices if cols is None else self.model.get_indices(cols)]

    def compute_prob_best_policies(
        self, tau: int = 0, cols: ColumnsType = None, tol: float = 1e-3, **kwargs: Any
    ) -> np.ndarray:
        """Compute the probability that a policy's rank is at most ``tau``.

        Args:
            tau (int, optional): Cutoff for the best policies. Defaults to 0.
            cols (ColumnsType, optional): Columns or policies of interest. Defaults to
                None.
            tol (float, optional): Precision of probability estimate. Defaults to 1e-3.

        Returns:
            np.ndarray: (# cols,) vector of probability estimates.

        Notes:
            The estimate for policy $i$, $\hat{p}_i$, has the property such that
            $Pr(r_i \leq \tau) \geq \hat{p}_i$ where $r_i$ is the true rank of policy
            $i$. The estimates are not designed to sum to 1.
        """

        def compute_prob_best_policies_col(i):
            # i is the index of a policy
            alpha = 0.5
            for j in range(1, iterations):
                adjustment = 1 / 2 ** (j + 1)
                alpha += (
                    adjustment
                    if self.conf_int(alpha, cols=[i], **kwargs)[0][1] > tau
                    else -adjustment
                )
            return alpha

        iterations = int(np.ceil(-np.log2(tol)))
        prob_best = 1 - np.array(
            [compute_prob_best_policies_col(i) for i in self.model.get_indices(cols)]
        )
        return prob_best[0] if len(prob_best) == 1 else prob_best


class MarginalResults(RankerResultsBase):
    """Results for marginal rank confidence interval estimates.

    Subclasses :class:`RankerResultsBase`.

    Args:
        model (Ranker): Ranker model.
        *args (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
        tails (str, optional): Specifices the tails to use when estimating the rankings.
            Can be "two", "lower", or "upper". Defaults to "two".
        n_samples (int, optional): Number of samples to use for approximation. Defaults
            to 10000.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ResultsBase`.

    Attributes:
        tails (str): Tails.
        delta_mean (list[np.ndarray]): ``delta_mean[i]`` is a (2 * # policies - 2,)
            array such that ``delta_mean[i][j]`` is the estimated difference between
            policies i and j for j < # policies and the estimated difference between
            policies j and i for j >= # policies.
        delta_cov (list[np.ndarray]): ``delta_cov[i]`` is the covariance matrix for
            ``delta_mean[i]``.
        rvs (list[np.ndarray]): ``rvs[i]`` is an (# samples, 2 * # policies - 2) array
            of draws for simultaneous confidence intervals based on ``delta_mean[i]`` and
            ``delta_cov[i]``.
    """

    TWO_TAILED, LOWER_TAILED, UPPER_TAILED = "two", "lower", "upper"

    @property
    def tails(self) -> str:
        return self._tails if hasattr(self, "_tails") else self.TWO_TAILED

    @tails.setter
    def tails(self, tails: str) -> None:
        self._tails = self._check_tails(tails)

    def __init__(
        self,
        model: Ranker,
        *args: Any,
        tails: str = "two",
        n_samples: int = 10000,
        title: str = "Marginal ranking estimates",
        **kwargs: Any,
    ):
        super().__init__(model, *args, title=title, **kwargs)
        self.params = (-self.model.mean).argsort().argsort()[self.indices]
        self.tails = tails

        self.delta_mean, self.delta_cov, self.rvs = [], [], []
        for i in range(len(model.mean)):
            delta_mean = self._compute_delta_mean(model.mean, i)
            self.delta_mean.append(np.concatenate([delta_mean, -delta_mean]))
            delta_cov = self._compute_delta_cov(model.cov, i, i)
            delta_cov = np.vstack(
                [np.hstack([delta_cov, -delta_cov]), np.hstack([-delta_cov, delta_cov])]
            )
            self.delta_cov.append(delta_cov)
            rvs = multivariate_normal.rvs(
                np.zeros(delta_cov.shape[0]), delta_cov, size=n_samples
            )
            self.rvs.append(rvs / np.sqrt(delta_cov.diagonal()))

    def _check_tails(self, tails: str) -> str:
        """Checks that the tails are valid.

        Args:
            tails (str): Tails.

        Raises:
            ValueError: ``tails`` must be "two", "lower", or "upper".

        Returns:
            str: Tails.
        """
        if tails is None:
            return self.tails

        if tails not in (None, self.TWO_TAILED, self.LOWER_TAILED, self.UPPER_TAILED):
            raise ValueError(
                f"tails must be one of {self.TWO_TAILED, self.LOWER_TAILED, self.UPPER_TAILED}, got {tails}."
            )

        return tails

    def compute_hypothesis_matrix(
        self, alpha: float = 0.05, tails: str = None, cols: ColumnsType = None
    ) -> pd.DataFrame:
        """Compute a hypothesis matrix.

        The hypothesis matrix is such that ``matrix[col][row]`` indicates whether the
        ``col`` policy is significantly better than the ``row`` policy.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            tails (str, optional): Tails to use. Defaults to None.
            cols (ColumnsType, optional): Columns for hypothesis testing. Defaults to
                None.

        Returns:
            pd.DataFrame: Hypothesis matrix.
        """
        tails = self._check_tails(tails)
        matrix = []
        for i, (delta_mean, delta_cov, rvs) in enumerate(
            zip(self.delta_mean, self.delta_cov, self.rvs)
        ):
            if tails != self.TWO_TAILED:
                # select the relevant objects for 1-tailed hypotheses
                indices = (
                    slice(0, self.n_policies - 1)
                    if tails == self.LOWER_TAILED
                    else slice(self.n_policies - 1, 2 * (self.n_policies - 1))
                )
                delta_mean = delta_mean[indices]
                delta_cov = delta_cov[indices, indices]
                rvs = rvs[:, indices]

            rejected = self._stepdown(delta_mean, delta_cov, rvs, alpha)[
                : len(self.model.mean) - 1
            ]
            rejected = np.insert(rejected, i, False)
            matrix.append(rejected)

        return self._select_hypothesis_matrix_columns(matrix, cols)

    def conf_int(
        self, alpha: float = 0.05, cols: ColumnsType = None, tails: str = None
    ) -> np.ndarray:
        """Compute confidence intervals for the policy ranking.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            cols (ColumnsType, optional): Columns for which to compute the ranking.
                Defaults to None.
            tails (str, optional): Tails. Defaults to None.

        Returns:
            np.ndarray: (# policies, 2) array of upper and lower bounds of the CI for
                each policy.
        """
        tails = self._check_tails(tails)
        conf_int = super().conf_int(alpha, cols, tails=tails)
        if tails == self.LOWER_TAILED:
            conf_int[:, 0] = 0
        elif tails == self.UPPER_TAILED:
            conf_int[:, 0] = self.n_policies - conf_int[:, 1] - 1
            conf_int[:, 1] = self.n_policies - 1
        return conf_int

    def compute_prob_best_policies(
        self, tau: int = 0, cols: ColumnsType = None, tol: float = 1e-3, **kwargs: Any
    ) -> np.ndarray:
        """Compute the probability that a policy's rank is at most ``tau``.

        See :class:`RankerResultsBase`.
        """
        return super().compute_prob_best_policies(
            tau, cols=cols, tol=tol, tails="lower", **kwargs
        )


class SimultaneousResults(RankerResultsBase):
    """Results for simultaneous rank confidence interval estimates.

    Subclasses :class:`RankerResultsBase`.

    Args:
        model (Ranker): Ranker model.
        *args (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
        tails (str, optional): Specifices the tails to use when estimating the rankings.
            Can be "two", "lower", or "upper". Defaults to "two".
        n_samples (int, optional): Number of samples to use for approximation. Defaults
            to 10000.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ResultsBase`.

    Attributes:
        delta_mean (np.ndarray): (# policies * (# policies - 1)) array of estimated
            differences between policies.
        delta_cov (np.ndarray):
            (# policies * (# policies - 1), # policies * (# policies  1)) covariance
            matrix of ``delta_mean``.
        rvs (np.ndarray): (# policies * (# policies - 1),) array of draws for computing
            simultaneous confidence intervals based on ``delta_mean`` and ``delta_cov``.
        test_stats (np.ndarray): (# policies, # policies - 1) matrix stuch that
            ``test_stats[tau, i]`` is the test statistic for policy i when computing a
            set of policies with rank <= tau.
    """

    def __init__(
        self,
        model: Ranker,
        *args: Any,
        n_samples: int = 10000,
        title: str = "Simultaneous ranking estimates",
        **kwargs: Any,
    ):
        super().__init__(model, *args, title=title, **kwargs)
        self.params = (-model.mean).argsort().argsort()[self.indices]

        delta_mean, delta_cov = [], []
        for i in range(len(model.mean)):
            delta_mean.append(self._compute_delta_mean(model.mean, i))
            delta_cov.append(
                np.hstack(
                    [
                        self._compute_delta_cov(model.cov, i, j)
                        for j in range(len(model.mean))
                    ]
                )
            )
        self.delta_mean = np.concatenate(delta_mean)
        self.delta_cov = np.vstack(delta_cov)
        self.rvs = multivariate_normal.rvs(
            np.zeros(len(self.delta_mean)), self.delta_cov, size=n_samples
        )
        self.rvs /= np.sqrt(self.delta_cov.diagonal())

        # compute test statistics
        arr = self.delta_mean / np.sqrt(self.delta_cov.diagonal())
        arr = arr.reshape((len(model.mean), len(model.mean) - 1))
        arr = np.hstack((arr, np.zeros((len(model.mean), 1))))
        arr.sort()
        self.test_stats = -arr.T

    def _get_mask(self, arr, get_mask_for_policy):
        if isinstance(arr, int):
            return get_mask_for_policy(arr)
        elif len(arr) == 0:
            return np.full(len(self.delta_mean), False)
        return np.array([get_mask_for_policy(i) for i in arr]).any(axis=0)

    def _get_i_minus_j_mask(self, arr: Union[int, np.ndarray]) -> np.ndarray:
        """Get a mask such that ``self.delta_mean[mask]`` are the estimated
        differences between a reference policy and another policy.

        Args:
            arr (Union[int, np.ndarray]): (n,) array of reference policies.

        Returns:
            np.ndarray: (# policies * (# policies - 1),) mask.
        """

        def get_mask_for_policy(i):
            indices = np.arange(
                i * (len(self.model.mean) - 1), (i + 1) * (len(self.model.mean) - 1)
            )
            mask = np.zeros(len(self.delta_mean))
            mask[indices] = 1
            return mask.astype(bool)

        return self._get_mask(arr, get_mask_for_policy)

    def _get_j_minus_i_mask(self, arr: Union[int, np.ndarray]) -> np.ndarray:
        """Get a mask such that ``self.delta_mean[mask]`` are the estimated
        differences between some policy and a reference policy.

        Args:
            arr (Union[int, np.ndarray]): Reference policies.

        Returns:
            np.ndarray: Mask.
        """

        def get_mask_for_policy(i):
            # delta_ji are the i-1'th indicies in groups before start and the i'th
            # indicies in the groups after end
            start, end = i * (len(self.model.mean) - 1), (i + 1) * (
                len(self.model.mean) - 1
            )
            indices = np.concatenate(
                (
                    np.arange(i - 1, start, len(self.model.mean) - 1),
                    np.arange(end + i, len(self.delta_mean), len(self.model.mean) - 1),
                )
            )
            if i == 0:
                # indices will start with -1, which is incorrect
                indices = indices[1:]
            mask = np.zeros(len(self.delta_mean))
            mask[indices] = 1
            return mask.astype(bool)

        return self._get_mask(arr, get_mask_for_policy)

    def compute_best_policies(
        self,
        tau: int = 0,
        alpha: float = 0.05,
        superset: bool = True,
        cols: ColumnsType = None,
    ) -> np.ndarray:
        """Compute a set of the ``tau``-best policies.

        Args:
            tau (int, optional): How many of the top policies to select. Defaults to 0.
            alpha (float, optional): Significance level. Defaults to 0.05.
            superset (bool, optional): Indicates that the returned set should be a
                superset of the true ``tau``-best policies with confidence
                1 - ``alpha``. If False, this will return a subset of the ``tau-best``
                policies. Defaults to True.
            cols (ColumnsType, optional): Columns to consider. Defaults to None.

        Returns:
            np.ndarray: (# columns,) boolean array indicating that policy i is in the
                selected set.
        """

        def compute_critical_value(subset):
            # compute the critical value to compare against test statistics
            # hypotheses with test statistics that exceed this critical value are rejected
            k_mask = ~self._get_i_minus_j_mask(subset)
            critical_value = np.quantile(
                self.rvs[:, k_mask & ~rejected_mask].max(axis=1), 1 - alpha
            )
            return critical_value

        indices = self.indices if cols is None else self.model.get_indices(cols)
        if tau == len(self.model.mean) - 1:
            return np.full(len(self.model.mean), True)[indices]

        if superset:
            test_stats = self.test_stats[tau]
        else:
            test_stats = -self.test_stats[tau + 1]
            tau = len(self.model.mean) - tau - 1

        rejected, newly_rejected = np.full(len(self.model.mean), False), None
        subsets = list(combinations(np.arange(len(self.model.mean)), tau))
        while newly_rejected is None or (newly_rejected.any() and (~rejected).any()):
            rejected_mask = self._get_j_minus_i_mask(np.where(rejected)[0])
            critical_value = max([compute_critical_value(i) for i in subsets])
            newly_rejected = (test_stats > critical_value) & ~rejected
            rejected = rejected | newly_rejected

        return (~rejected if superset else rejected)[indices]

    def compute_hypothesis_matrix(
        self, alpha: float = 0.05, cols: ColumnsType = None
    ) -> pd.DataFrame:
        """Compute a hypothesis matrix.

        The hypothesis matrix is such that ``matrix[col][row]`` indicates whether the
        ``col`` policy is significantly better than the ``row`` policy.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            cols (ColumnsType, optional): Columns for hypothesis testing. Defaults to
                None.

        Returns:
            pd.DataFrame: Hypothesis matrix.
        """
        matrix = []
        rejected = self._stepdown(self.delta_mean, self.delta_cov, self.rvs, alpha)
        for i in range(len(self.model.mean)):
            rejected_i = rejected[self._get_i_minus_j_mask(i)]
            rejected_i = np.insert(rejected_i, i, False)
            matrix.append(rejected_i)

        return self._select_hypothesis_matrix_columns(matrix, cols)

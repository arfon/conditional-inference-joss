"""Simultaneous confidence sets and multiple hypothesis testing.

References:

    .. code-block::

        @article{romano2005stepwise,
            title={Stepwise multiple testing as formalized data snooping},
            author={Romano, Joseph P and Wolf, Michael},
            journal={Econometrica},
            volume={73},
            number={4},
            pages={1237--1282},
            year={2005},
            publisher={Wiley Online Library}
        }

        @techreport{mogstad2020inference,
            title={Inference for ranks with applications to mobility across neighborhoods and academic achievement across countries},
            author={Mogstad, Magne and Romano, Joseph P and Shaikh, Azeem and Wilhelm, Daniel},
            year={2020},
            institution={National Bureau of Economic Research}
        }
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import multivariate_normal, norm

from conditional_inference.base import ColumnsType, ModelBase, ResultsBase


class ConfidenceSetResults(ResultsBase):
    """Results for simultaneous confidence sets.

    Subclasses :class:`conditional_inference.base.ResultsBase`.

    Args:
        n_samples (int, optional): Number of samples to draw when approximating the
            confidence set. Defaults to 10000.
    """

    _default_title = "Confidence set results"

    def __init__(self, *args: Any, n_samples: int = 10000, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.params = self.model.mean.copy()

        # draw random values for confidence set approximation
        mean = np.zeros(2 * len(self.model.mean))  # (2 * # params,)
        cov = np.vstack(
            [
                np.hstack([self.model.cov, -self.model.cov]),
                np.hstack([-self.model.cov, self.model.cov]),
            ]
        )  # (2 * # params, 2 * # params)
        self._std_diagonal = np.sqrt(cov.diagonal())  # (2 * # params,)
        self._rvs = multivariate_normal.rvs(
            mean,
            cov,
            size=n_samples,
            random_state=self.model.random_state,
        )  # (# samples, 2 * # params)
        self._rvs /= self._std_diagonal

        if self.model.n_params == 1:
            self.pvalues = 2 * np.atleast_1d(
                norm.cdf(-abs(self.model.mean[0]), 0, np.sqrt(self.model.cov[0, 0]))
            )
        else:
            params = self.params.reshape(-1, 1).repeat(n_samples, axis=1)
            arr = self._rvs.max(axis=1) * np.sqrt(self.model.cov.diagonal()).reshape(
                -1, 1
            ).repeat(n_samples, axis=1)
            self.pvalues = np.array(
                [(params - arr < 0).mean(axis=1), (params + arr > 0).mean(axis=1)]
            ).min(axis=0)

    def _conf_int(self, alpha: float, indices: np.ndarray) -> np.ndarray:
        if self.model.n_params == 1:
            return np.atleast_2d(
                norm.ppf(
                    [alpha / 2, 1 - alpha / 2],
                    self.model.mean[0],
                    np.sqrt(self.model.cov[0, 0]),
                )
            )

        params = self.params[indices]
        arr = (
            np.quantile(self._rvs.max(axis=1), 1 - alpha) * self._std_diagonal[indices]
        )
        return np.array([params - arr, params + arr]).T

    def test_hypotheses(
        self, alpha: float = 0.05, columns: ColumnsType = None
    ) -> pd.DataFrame:
        """Test the null hypothesis that the parameter is equal to 0.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            columns (ColumnsType, optional): Selected columns. Defaults to None.

        Returns:
            pd.DataFrame: Results dataframe.
        """
        params = np.concatenate([self.params, -self.params])

        rejected, newly_rejected = np.full(self._rvs.shape[1], False), None
        while newly_rejected is None or (newly_rejected.any() and not rejected.all()):
            quantile = np.quantile(self._rvs[:, ~rejected].max(axis=1), 1 - alpha)
            newly_rejected = (params - quantile * self._std_diagonal > 0) & ~rejected
            rejected = rejected | newly_rejected

        indices = self.model.get_indices(columns)
        return pd.DataFrame(
            rejected.reshape(2, -1).T[indices],
            columns=["param>0", "param<0"],
            index=self.model.exog_names[indices],
        )

    def _make_summary_header(self, alpha: float) -> list[str]:
        return [
            "coef (conventional)",
            "pvalue",
            f"{1-alpha} CI lower",
            f"{1-alpha} CI upper",
        ]


class ConfidenceSet(ModelBase):
    """Model for simultaneous confidence sets.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.confidence_set import ConfidenceSet

            x = np.arange(-1, 2)
            cov = np.identity(3) / 10
            model = ConfidenceSet(x, cov)
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                              Confidence set results
            =========================================================
               coef (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------
            x0              -1.000  0.004        -1.762        -0.238
            x1               0.000  1.000        -0.762         0.762
            x2               1.000  0.004         0.238         1.762
            ===============
            Dep. Variable y
            ---------------

        .. testcode::

            print(results.test_hypotheses())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                param>0  param<0
            x0    False     True
            x1    False    False
            x2     True    False
    """

    _results_cls = ConfidenceSetResults


class AverageComparison(ConfidenceSet):
    """Compare each parameter to the average value across all parameters.

    Subclasses :class:`ConfidenceSet`.

    Args:
        *args (Any): Passed to :class:`ConfidenceSet`.
        **kwargs (Any): Passed to :class:`ConfidenceSet`.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.confidence_set import AverageComparison

            x = np.arange(-1, 2)
            cov = np.identity(3) / 10
            model = AverageComparison(x, cov)
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                              Confidence set results
            =========================================================
               coef (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------
            x0              -1.000  0.000        -1.607        -0.393
            x1               0.000  1.000        -0.607         0.607
            x2               1.000  0.000         0.393         1.607
            ===============
            Dep. Variable y
            ---------------
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ones = np.ones((len(self.mean), 1))
        identity = np.identity(len(self.mean))
        cov_inv = np.linalg.inv(self.cov)
        projection = ones @ np.linalg.inv(ones.T @ cov_inv @ ones) @ ones.T @ cov_inv
        self.mean = (identity - projection) @ self.mean
        self.cov = (identity - projection) @ self.cov @ (identity - projection).T


def _compute_delta_mean(mean: np.ndarray, i: int) -> np.ndarray:
    """Computes the difference between each estimated parameter and a baseline.

    Args:
        mean (np.ndarray): (# params,) array of estimated parameters.
        i (int): Index of the baseline parameter.

    Returns:
        np.ndarray: (# params - 1,) array of differences.
    """
    return np.delete(mean - mean[i], i)


def _get_delta_names(names: np.ndarray, i: int) -> np.ndarray:
    """Get names for the parameter differences, e.g., "x0 - x1".

    Args:
        names (np.ndarray): Original parameter names.
        i (int): Index of the baseline parameter.

    Returns:
        np.ndarray: (# params - 1,) array of names of the differences.
    """
    return np.delete([f"{name} - {names[i]}" for name in names], i)


def _compute_delta_cov(cov: np.ndarray, i: int, j: int = None) -> np.ndarray:
    """Compute the covariance of (mean - mean[i], mean - mean[j]).

    Args:
        cov (np.ndarray): Covariance of original parameter estimates.
        i (int): Baseline parameter.
        j (int, optional): Baseline parameter. Defaults to None.

    Returns:
        np.ndarray: (# params - 1, # params - 1) covariance matrix of differences.
    """
    if j is None:
        j = i
    repeat_i = np.repeat(np.atleast_2d(cov[i]), cov.shape[0], axis=0)
    repeat_j = np.repeat(np.atleast_2d(cov[j]), cov.shape[0], axis=0).T
    delta_cov = cov[i, j] + cov - repeat_i - repeat_j
    return np.delete(np.delete(delta_cov, i, axis=0), j, axis=1)


class BaselineComparison(ConfidenceSet):
    """Compare parameters to a baseline parameter.

    Subclasses :class:`ConfidenceSet`.

    Args:
        baseline (Union[int, str]): Index or name of the baseline parameter.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.confidence_set import BaselineComparison

            x = np.arange(-1, 2)
            cov = np.identity(3) / 10
            model = BaselineComparison(x, cov, exog_names=["x0", "x1", "x2"], baseline="x0")
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                              Confidence set results
            =========================================================
               coef (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------
            x1               1.000  0.046         0.021         1.979
            x2               2.000  0.000         1.021         2.979
            ===============
            Dep. Variable y
            ---------------
    """

    def __init__(self, *args, baseline: Union[int, str], **kwargs):
        super().__init__(*args, **kwargs)
        index = int(
            baseline
            if isinstance(baseline, (float, int))
            else list(self.exog_names).index(baseline)
        )
        self.mean = _compute_delta_mean(self.mean, index)
        self.cov = _compute_delta_cov(self.cov, index)
        if self._exog_names is not None:
            self.exog_names = np.delete(self.exog_names, index)


class PairwiseComparisonResults(ConfidenceSetResults):
    """Results of pairwise comparisons.

    Subclasses :class:`ConfidenceSetResults`.
    """

    _default_title = "Pairwise comparisons"

    def test_hypotheses(
        self, alpha: float = 0.05, columns: ColumnsType = None, wide: bool = True
    ) -> pd.DataFrame:
        """Test pairwise hypotheses.

        Args:
            alpha (float, optional): Significance level. Defaults to .05.
            columns (ColumnsType, optional): Selected columns. In wide format, these are
                the original column names (e.g., "x0"). In long format, these are the
                names of the differences (e.g., "x1 - x0"). Defaults to None.
            wide (bool, optional): Return the results is wide (square) format. Defaults
                to True.

        Returns:
            pd.DataFrame: Results.
        """
        if not wide:
            return super().test_hypotheses(alpha, columns)

        # reshape rejected dataframe into a triangular matrix
        rejected = super().test_hypotheses(alpha).values
        tri = np.full((self.model.n_params, self.model.n_params), False)
        indices = np.triu_indices(self.model.n_params, 1)
        tri[indices] = rejected[:, 0]
        tri[(indices[1], indices[0])] = rejected[:, 1]

        indices = self.model.get_indices(columns, self.model.exog_names_orig)
        column_names = self.model.exog_names_orig[indices]
        return pd.DataFrame(
            tri[indices][:, indices], index=column_names, columns=column_names
        )

    def hypothesis_heatmap(
        self,
        *args: Any,
        title: str = None,
        ax=None,
        triangular: bool = False,
        **kwargs: Any,
    ):
        """Create a heatmap of pairwise hypothesis tests.

        Args:
            title (str, optional): Title.
            ax (AxesSubplot, optional): Axis to write on. Defaults to None.
            triangular (bool, optional): Display the results in a triangular (as opposed
                to square) output. Usually, you should set this to True if and only if
                your columns are sorted. Defaults to False.

        Returns:
            AxesSubplot: Plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        matrix = self.test_hypotheses(*args, **kwargs)
        if triangular:
            mask = np.zeros_like(matrix)
            mask[np.triu_indices_from(mask)] = True
        else:
            mask = None

        sns.heatmap(
            matrix,
            cbar=False,
            ax=ax,
            yticklabels=matrix.index,
            xticklabels=matrix.columns,
            mask=mask,
            square=True,
            cmap=sns.color_palette()[3:1:-1],
            center=0.5,
        )
        ax.set_title(title or self.title)
        plt.yticks(rotation=0)
        return ax

    def _make_summary_header(self, alpha: float) -> list[str]:
        return [
            "delta (conventional)",
            "pvalue",
            f"{1-alpha} CI lower",
            f"{1-alpha} CI upper",
        ]


class PairwiseComparison(ConfidenceSet):
    """Compute pairwise comparisons.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.confidence_set import PairwiseComparison

            x = np.arange(-1, 2)
            cov = np.identity(3) / 10
            model = PairwiseComparison(x, cov)
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                                  Pairwise comparisons
            ===============================================================
                    delta (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------------
            x1 - x0                1.000  0.067        -0.045         2.045
            x2 - x0                2.000  0.000         0.955         3.045
            x2 - x1                1.000  0.067        -0.045         2.045
            ===============
            Dep. Variable y
            ---------------

        .. testcode::

            print(results.test_hypotheses())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                   x0     x1     x2
            x0  False  False   True
            x1  False  False  False
            x2  False  False  False

        This means that parameter x2 is significantly greater than x0.
    """

    _results_cls = PairwiseComparisonResults

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.exog_names_orig = self.exog_names
        self.exog_names = np.concatenate(
            [_get_delta_names(self.exog_names, i)[i:] for i in range(self.n_params)]
        )
        self.mean = np.concatenate(
            [_compute_delta_mean(self.mean, i)[i:] for i in range(self.n_params)]
        )
        self.cov = np.vstack(
            [
                np.hstack(
                    [
                        _compute_delta_cov(self.cov, i, j)[i:, j:]
                        for j in range(self.n_params)
                    ]
                )
                for i in range(self.n_params)
            ]
        )


class MarginalRankingResults(ResultsBase):
    """Marginal ranking results."""

    _default_title = "Marginal ranking"

    def __init__(self, model: MarginalRanking, *args: Any, **kwargs: Any):
        super().__init__(model, *args, **kwargs)
        self.params = (-self.model.mean).argsort().argsort() + 1
        self._baseline_comparisons = [
            BaselineComparison(model.mean, model.cov, baseline=i).fit()
            for i in range(model.n_params)
        ]

    def _conf_int(self, alpha: float, indices: np.ndarray) -> np.array:
        def get_rank_ci(results):
            hypotheses_count = results.test_hypotheses(alpha).sum(axis=0)
            return [hypotheses_count[0], self.model.n_params - hypotheses_count[1] - 1]

        return (
            np.array([get_rank_ci(self._baseline_comparisons[i]) for i in indices]) + 1
        )

    def _make_summary_header(self, alpha: float) -> list[str]:
        return [
            "rank (conventional)",
            "pvalue",
            f"{1-alpha} CI lower",
            f"{1-alpha} CI upper",
        ]


class MarginalRanking(ConfidenceSet):
    """Estimate rankings with marginal confidence intervals.

    Subclasses :class:`ConfidenceSet`.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.confidence_set import MarginalRanking

            x = np.arange(-1, 2)
            cov = np.diag([1, 2, 3]) / 10
            model = MarginalRanking(x, cov)
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                              Marginal ranking
            =========================================================
               rank (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------
            x0               3.000    nan         2.000         3.000
            x1               2.000    nan         1.000         3.000
            x2               1.000    nan         1.000         2.000
            ===============
            Dep. Variable y
            ---------------

    """

    _results_cls = MarginalRankingResults


class SimultaneousRankingResults(ResultsBase):
    """Simultaneous ranking results."""

    _default_title = "Simultaneous ranking"

    def __init__(self, model: SimultaneousRanking, *args: Any, **kwargs: Any):
        super().__init__(model, *args, **kwargs)
        self.params = (-model.mean).argsort().argsort() + 1
        pairwise_model = PairwiseComparison(model.mean, model.cov)
        self._pairwise_comparison = pairwise_model.fit()

        # compute test statistics for finding the top tau parameters
        # self._test_stats is a (# params, # params) matrix where
        # `self._test_stats[tau, k]`` is the test statistic for the null hypothesis that
        # the parameter k is not in the top tau parameters
        indices = np.triu_indices(model.n_params, 1)
        self._test_stats = np.full((model.n_params, model.n_params), 0.0)
        test_stats = pairwise_model.mean / np.sqrt(pairwise_model.cov.diagonal())
        self._test_stats[indices] = -test_stats
        self._test_stats[(indices[1], indices[0])] = test_stats
        self._test_stats = np.sort(self._test_stats, 0)[::-1]

        # compute random values to find the critical values for finding the top tau
        # parameters
        # self._rvs is a (# samples, # params, # params) matrix where
        # `self._rvs[n, k, l]`` is the nth sample of the studentized param_k - param_l
        def reshape(arr):
            arr = arr[: int(len(arr) / 2)]
            matrix = np.zeros((model.n_params, model.n_params))
            matrix[indices] = arr
            matrix[(indices[1], indices[0])] = -arr
            return matrix

        self._rvs = np.apply_along_axis(reshape, -1, self._pairwise_comparison._rvs)

    def _conf_int(self, alpha: float, indices: np.ndarray) -> np.ndarray:
        hypothesis_matrix = self._pairwise_comparison.test_hypotheses(alpha).values
        return (
            np.array(
                [
                    hypothesis_matrix.sum(axis=1),
                    self.model.n_params - hypothesis_matrix.sum(axis=0) - 1,
                ]
            ).T[indices]
            + 1
        )

    def compute_best_params(
        self, n_best_params: int = 1, alpha: float = 0.05, superset: bool = True
    ) -> pd.Series:
        """Compute the set of best (largest) parameters.

        Find the set of parameters such that the truly best ``n_best_params`` parameters
        are in this set with probability ``1-alpha``. Or, find the set of parameters
        such that these parameters are in the truly best ``n_best_params`` parameters
        with probability ``1-alpha``.

        Args:
            n_best_params (int, optional): Number of best parameters. Defaults to 1.
            alpha (float, optional): Significance level. Defaults to 0.05.
            superset (bool, optional): Indicates that the returned set is a superset of
                the truly best n parameters. If False, the returned set is a subset of
                the truly best n parameters. Defaults to True.

        Returns:
            pd.Series: Indicates which parameters are in the selected set.
        """
        if superset:
            test_stats = self._test_stats[n_best_params - 1]
        else:
            test_stats = -self._test_stats[n_best_params]
            n_best_params = self.model.n_params - n_best_params

        subsets = []
        for indices in combinations(np.arange(self.model.n_params), n_best_params - 1):
            arr = np.full(self.model.n_params, True)
            if len(indices) > 0:
                arr[list(indices)] = False

            subsets.append(arr)

        compute_critical_value = lambda subset: np.quantile(
            rvs[:, :, subset].max(axis=(1, 2)), 1 - alpha
        )
        rejected, newly_rejected = np.full(self.model.n_params, False), None
        while newly_rejected is None or (newly_rejected.any() and not rejected.all()):
            rvs = self._rvs[:, ~rejected]
            critical_value = max([compute_critical_value(subset) for subset in subsets])
            newly_rejected = (test_stats > critical_value) & ~rejected
            rejected = newly_rejected | rejected

        return pd.Series(
            ~rejected if superset else rejected, index=self.model.exog_names
        )

    def _make_summary_header(self, alpha: float) -> list[str]:
        return [
            "rank (conventional)",
            "pvalue",
            f"{1-alpha} CI lower",
            f"{1-alpha} CI upper",
        ]


class SimultaneousRanking(ConfidenceSet):
    """Estimate rankings with simultaneous confidence intervals.

    Subclasses :class:`ConfidenceSet`.

    Examples:

        .. testcode::

            import numpy as np
            from conditional_inference.confidence_set import SimultaneousRanking

            x = np.arange(3)
            cov = np.identity(3) / 10
            model = SimultaneousRanking(x, cov)
            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

                               Simultaneous ranking
            =========================================================
               rank (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------
            x0               3.000    nan         2.000         3.000
            x1               2.000    nan         1.000         3.000
            x2               1.000    nan         1.000         2.000
            ===============
            Dep. Variable y
            ---------------

        .. testcode::

            print(results.compute_best_params())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

            x0    False
            x1    False
            x2     True
            dtype: bool

        This we can be 95% confident that the best (largest) parameter is x2.
    """

    _results_cls = SimultaneousRankingResults

"""Quantile-unbiased estimation
"""
from __future__ import annotations

from typing import Any, List, Sequence, Union

import numpy as np
from scipy.stats import multivariate_normal
from statsmodels.iolib.summary import Summary

from .base import (
    ConventionalEstimatesData,
    ModelBase,
    ResultsBase,
    ColumnType,
    ColumnsType,
    Numeric1DArray,
)
from .stats import quantile_unbiased


class RQUData(ConventionalEstimatesData):
    """Ranked quantile-unbiased estimator data.

    Args:
        mean (Numeric1DArray): (n,) array of conventional estimates used to rank-order
            policies.
        cov (np.ndarray): (n,n) covariance matrix of ``mean``.
        endog_names (Union[str, Sequence[str]], optional): Names of endogenous
            variables. Defaults to None.
        exog_names (Sequence[str], optional): (n,) sequence of names of exogenous
            variables (i.e., the policies). Defaults to None.
        ymean (Numeric1DArray, optional): (n,) conventional estimates of policy
            effects. Defaults to None.
        ycov (np.ndarray, optional): (n,n) covariance matrix of ``ymean``. Defaults to
            None.
        xycov (np.ndarray, optional): (n,n) covariance matrix of ``mean`` and ``ymean``.
            Defaults to None.

    Attributes:
        mean (np.ndarray): (n,) array of conventional estimates used to rank-order
            policies/
        cov (np.ndarray): (n, n) covariance matrix of ``mean``.
        endog_names (str): Name of the endogenous variable.
        exog_names (Sequence[str]): (n,) sequence of names of exogenous variables
            (i.e., the policies).
        ymean (np.ndarray): (n,) conventional estimates of policy effects. If ``None``,
            ``mean`` are assumed to be the policy effects.
        ycov (np.ndarray): (n, n) covariance matrix of ``ymean``. If ``None``, use
            ``cov``.
        xycov (np.ndarray): (n,n) covariance matrix of ``mean`` and ``ymean``. If
            ``None``, use ``cov``.

    Note:

        By default, we assume that the conventional estimates used to rank policies are
        the same as the conventional estimates of the policy effects. If this is not the
        case, set ``mean`` and ``cov`` to the conventional estimates used to rank the
        policies and ``ymean`` and ``ycov`` to the conventional estimates of the policy
        effects. You must also set ``xycov`` to the covariance matrix of ``mean`` and
        ``ymean``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        endog_names: str = None,
        exog_names: Union[str, Sequence[str]] = None,
        ymean: Numeric1DArray = None,
        ycov: np.ndarray = None,
        xycov: np.ndarray = None,
    ):
        super().__init__(mean, cov, endog_names, exog_names)
        self.ymean = ymean
        self.ycov = ycov
        self.xycov = xycov

    @property
    def ymean(self):  # pylint: disable=missing-function-docstring
        return self.mean if self._ymean is None else self._ymean

    @ymean.setter
    def ymean(self, ymean):  # pylint: disable=missing-function-docstring
        self._ymean = None if ymean is None else np.atleast_1d(ymean)

    @property
    def ycov(self):  # pylint: disable=missing-function-docstring
        return self.cov if self._ycov is None else self._ycov

    @ycov.setter
    def ycov(self, ycov):  # pylint: disable=missing-function-docstring
        self._ycov = ycov

    @property
    def xycov(self):  # pylint: disable=missing-function-docstring
        return self.cov if self._xycov is None else self.xycov

    @xycov.setter
    def xycov(self, xycov):  # pylint: disable=missing-function-docstring
        self._xycov = xycov


class RQU(ModelBase):
    """Ranked quantile-unbiased estimator.

    Provides utilities for obtaining quantile-unbiased estimates conditional on the
        rank-ordering of conventional estimates of policy effects.

    Args:
        mean (Numeric1DArray): (n,) array of conventional estimates of policy effects.
        cov (np.ndarray): (n,n) covariance matrix of ``mean``.
        seed (int, optional): Random seed. Defaults to 0.
        **args (Any): Additional arguments passed to :class:`RQUData` constructor.
        **kwargs (Any): Additional keyword arguments passed to :class:`RQUData`
            constructor.

    Attributes:
        data (RQUData): Ranked quantile-unbiased estimator data.
        seed (int): Random seed.

    You can set and access ``self.data`` attributes directly, e.g.,

    .. testsetup::

        from conditional_inference.rqu import RQU
        import numpy as np

    .. doctest::

        >>> rqu = RQU(mean=np.arange(3), cov=np.identity(3))
        >>> rqu.mean
        array([0, 1, 2])
    """

    _data_properties = [
        "mean",
        "cov",
        "endog_names",
        "exog_names",
        "ymean",
        "ycov",
        "xycov",
    ]

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        *args: Any,
        seed: int = 0,
        **kwargs: Any,
    ):
        self.seed = seed
        self.data = RQUData(mean, cov, *args, **kwargs)

    def compute_projection_quantile(
        self, alpha: float = 0.05, n_samples: int = 10000
    ) -> float:
        """Compute the 1-alpha quantile for projection confidence intervals.

        Args:
            alpha (float, optional): Quantile level of the projection CI. Defaults to
                0.05.
            n_samples (int, optional): Number of samples used in approximating the
                1-alpha quantile. Defaults to 10000.

        Returns:
            float: 1-alpha quantile of the projection CI.
        """
        if alpha == 0:
            return np.inf
        rvs = self.projection_rvs(size=n_samples)
        return np.quantile(abs(rvs).max(axis=1), 1 - alpha)

    def fit(
        self,
        cols: ColumnsType = None,
        projection: bool = False,
        **kwargs: Any,
    ) -> Union[ProjectionResults, RQUResults]:
        """Fit the RQU estimator and return results.

        Args:
            cols (ColumnsType, optional): Names or indices of the policies of interest.
                Defaults to None.
            projection (bool, optional): If True, return projection results. If False,
                return quantile-unbiased results. Defaults to False.

        Returns:
            Union[ProjectionResults, RQUResults]: Quantile-unbiased estimation results.

        Examples:

            Suppose we have 5 policies, each with a true effect of 0. The observed
            effect of the policies is sampled from a joint normal with identity
            covariance matrix.

            .. code-block:: python

                >>> from conditional_inference.rqu import RQU
                >>> import numpy as np
                >>> npolicies = 5
                >>> mean = np.random.normal(size=npolicies)
                >>> cov = np.identity(npolicies)
                >>> rqu = RQU(mean, cov)
                >>> results = rqu.fit(cols="sorted", beta=.005)
                >>> print(results.summary())
                 Conditional quantile-unbiased estimates
                =====================================
                   coef (median) pvalue [0.025 0.975]
                -------------------------------------
                x1         0.388  0.412 -2.209  2.931
                x2         0.487  0.413 -2.885  3.672
                x4        -1.289  0.700 -4.354  2.590
                x0        -1.468  0.664 -4.775  2.690
                x3        -0.154  0.529 -3.316  2.183
                ===============
                Dep. Variable y
                ---------------
        """
        if projection:
            return ProjectionResults(self, cols, **kwargs)
        return RQUResults(self, cols, **kwargs)

    def get_distribution(  # pylint: disable=too-many-arguments
        self,
        col: ColumnType = None,
        rank: Union[str, int] = "exact",
        beta: float = 0,
        n_samples: int = 10000,
        **kwargs: Any,
    ) -> quantile_unbiased:
        """Compute a quantile-unbiased distribution of the average effect of a policy.

        Args:
            col (ColumnType, optional): Name or index of the policy of interest.
                Defaults to None.
            rank (Union[str, int], optional): Rank of the policy of interest. The
                "exact" condition means that we condition on the policy we observed to
                be the best was in fact observed to be the best. Defaults to "exact".
            beta (float, optional): Projection quantile for hybrid estimation. Defaults
                to 0.
            n_samples (int, optional): Number of samples used to approximate the
                projection confidence interval. Defaults to 10000.
            **kwargs (Any): Additional keyword arguments are passed to the
                :class:`quantile_unbiased` constructor.

        Returns:
            quantile_unbiased: Quantile-unbiased distribution of the policy effect.
        """

        def get_index_rank(col, rank):
            # return the index and valid rank order(s) of the policy of interest
            if isinstance(rank, str) and rank not in ("exact", "floor", "ceil"):
                raise ValueError(
                    f"If `rank` is a string, must be 'exact', 'floor', or 'ceil', (got {rank})"
                )

            if col is None:
                if rank == "exact":
                    rank = 0
                if not isinstance(rank, int):
                    raise ValueError(
                        f"If `col` is not specified, `rank` must be 'exact' or int (got {rank})."
                    )
                index = np.argsort(-self.mean)[rank]
            else:
                index = self._get_index(col)
                exact_rank = (self.mean > self.mean[col]).sum()
                if isinstance(rank, str):
                    if rank == "exact":
                        rank = exact_rank
                    elif rank == "floor":
                        rank = np.arange(exact_rank + 1)
                    elif rank == "ceil":
                        rank = np.arange(exact_rank, self.mean.shape[0])

            return index, np.atleast_1d(rank) % self.mean.shape[0]

        def check_s_V_condition():  # pylint: disable=invalid-name
            # check that condition on set V is satisifed
            # V is the set of parameters with X-Y covariances equal to that of the
            # target parameter
            s_v = self.xycov[i, i] == self.xycov[:, i]
            if (-(z - z[i]))[s_v].min() < 0:
                indices = np.arange(self.ymean.shape[0])
                invalid_indices = np.where(s_v & (indices != i))[0]
                raise ValueError(
                    " ".join(
                        [
                            f"Empty truncation set for index {i} and rank {rank}.",
                            f"Parameters at indices {invalid_indices.tolist()} have equal X-Y",
                            f"covariances with the parameter at target index {i}.",
                        ]
                    )
                )

        def compute_truncation_set():
            # compute the trucation set for `truncnorm.cdf`
            # see paper for details on this algorithm

            def update_truncation_set(idx, j):
                if (
                    (tau_upper_size - tau_any_size <= rank)
                    & (rank <= tau_upper_size + tau_any_size)
                ).any():
                    if j is None:
                        # no possible upper bounding parameters => upper bound is np.inf
                        truncset.append((q[order[0]], np.inf))
                    elif j == order[-1]:
                        # no possible lower bounding parameters => lower bound is -np.inf
                        truncset.append((-np.inf, q[j]))
                    else:
                        truncset.append((q[order[idx + 1]], q[j]))

            # parameters which are eligible to serve as upper or lower bounds
            theta = self.xycov[i, i] != self.xycov[:, i]
            # possible threshold values
            q = (  # pylint: disable=invalid-name
                self.ycov[i, i]
                * (z[theta] - z[i])
                / (self.xycov[i, i] - self.xycov[theta, i])
            )
            order = np.argsort(-q)
            # indicates parameters which beat i
            # when the set of possible upper bounding parameters is empty
            tau_upper = self.xycov[i, i] < self.xycov[theta, i]
            # number of parameters which beat i
            # when the set of possible upper bounding parameters is empty
            tau_upper_size = tau_upper.sum()
            # number of parameters which could beat i
            # regardless of the upper and lower bounding parameters
            tau_any_size = (self.xycov[i, i] == self.xycov[:, i]).sum() - 1

            # compute the truncation set
            truncset = []
            update_truncation_set(None, None)
            for idx, j in enumerate(order):
                # update the size of winning parameters when theta_j moves
                # from possible lower bounding parameters
                # to possible upper bounding parameters
                tau_upper_size -= 1 if tau_upper[j] else -1
                update_truncation_set(idx, j)

            return truncset

        i, rank = get_index_rank(col, rank)
        z = (  # pylint: disable=invalid-name
            self.mean - (self.xycov[:, i] / self.ycov[i, i]) * self.ymean[i]
        )
        check_s_V_condition()
        if beta != 0:
            kwargs["projection_interval"] = self.compute_projection_quantile(
                beta, n_samples
            ) * np.sqrt(self.ycov[i, i])
        return quantile_unbiased(  # type: ignore
            y=self.ymean[i],
            scale=np.sqrt(self.ycov[i, i]),
            truncation_set=compute_truncation_set(),
            **kwargs,
        )

    def get_distributions(
        self,
        cols: ColumnsType = None,
        beta: float = 0,
        n_samples: int = 10000,
        **kwargs: Any,
    ) -> List[quantile_unbiased]:
        """Compute quantile-unbiased distributions of average policy effects.

        Args:
            cols (ColumnsType, optional): Names or indices of policies of interest.
                Defaults to None.
            beta (float, optional): Projection quantile for hybrid estimation. Defaults
                to 0.
            n_samples (int, optional): Number of samples used to approximate projection
                confidence intervals. Defaults to 10000.
            **kwargs (Any): Additional keyword arguments are passed to
                :meth:`RQU.get_distribution`.

        Returns:
            List[quantile_unbiased]: Quantile-unbiased distributions of policy effects.
        """
        if beta == 0:
            return [self.get_distribution(i, **kwargs) for i in self.get_indices(cols)]
        projection_intervals = self.compute_projection_quantile(
            beta, n_samples
        ) * np.sqrt(self.ycov.diagonal())
        return [
            self.get_distribution(i, projection_interval=interval, **kwargs)
            for i, interval in zip(self.get_indices(cols), projection_intervals)
        ]

    def projection_rvs(self, size: int = 1) -> np.ndarray:
        """Sample random values to construct projection confidence intervals.

        Args:
            size (int, optional): Number of samples. Defaults to 1.

        Returns:
            np.ndarray: (size, 2) array of samples.
        """
        rvs = multivariate_normal.rvs(
            np.zeros(self.ymean.shape), self.ycov, size=size, random_state=self.seed
        )
        rvs /= np.sqrt(self.ycov.diagonal())
        return np.array([rvs.min(axis=1), rvs.max(axis=1)]).T


class ProjectionResults(ResultsBase):
    """Projection confidence interval results.

    Projection confidence intervals have unconditionally correct coverage.

    Args:
        model (RQU): The RQU model instance.
        cols (ColumnsType, optional): Names or indices of policies of interest.
            Defaults to None.
        n_samples (int, optional): Number of samples used to approximate projection
            confidence intervals. Defaults to 10000.
        title (str, optional): Results title. Defaults to "Projection estimates".

    Attributes:
        model (RQU): The model instance.
        indices (List[int]): Indices of the policies of interest.
        params (np.ndarray): (n,) array of conventional point estimates.
        projection_rvs (np.ndarray): (n_samples, 2) array of samples used to construct
            projection CIs.
        pvalues (np.ndarray): (n,) array of probabilities that the true effect of a
            policy is less than 0.
        std_params_diag (np.ndarray): (n,) array of standard deviations from the
            ``mean`` covariance matrix.

    Examples:

        .. code-block:: python

            >>> from conditional_inference.rqu import RQU
            >>> import numpy as np
            >>> npolicies = 5
            >>> mean = np.random.normal(size=npolicies)
            >>> cov = np.identity(npolicies)
            >>> rqu = RQU(mean, cov)
            >>> results = rqu.fit(cols="sorted", projection=True)
            >>> print(results.summary())
                                 Projection estimates
            =========================================================
               coef (conventional) pvalue 0.95 CI lower 0.95 CI upper
            ---------------------------------------------------------
            x0               1.644  0.233        -0.936         4.223
            x2               0.813  0.693        -1.766         3.393
            x3               0.217  0.931        -2.362         2.796
            x1               0.060  0.962        -2.519         2.639
            x4              -0.064  0.976        -2.643         2.515
            ===============
            Dep. Variable y
            ---------------

    """

    def __init__(
        self,
        model: RQU,
        cols: ColumnsType = None,
        n_samples: int = 10000,
        title: str = "Projection estimates",
    ):
        def compute_pvalues():
            params = self.params.reshape(-1, 1).repeat(n_samples, axis=1)
            std = self.std_params_diag.reshape(-1, 1).repeat(n_samples, axis=1)
            arr = params + self.projection_rvs[:, 0] * std
            return (arr < 0).mean(axis=1)

        super().__init__(model, cols, title)
        self.params = model.ymean[self.indices]
        self.projection_rvs = model.projection_rvs(n_samples)
        self.std_params_diag = np.sqrt(model.ycov.diagonal())[self.indices]
        self.pvalues = compute_pvalues()

    def conf_int(self, alpha: float = 0.05, cols: ColumnsType = None) -> np.ndarray:
        """Compute the 1-alpha confidence interval.

        Args:
            alpha (float, optional): The CI will cover the truth with probability
                greater than 1-alpha. Defaults to 0.05.
            cols (ColumnsType, optional): Names or indices of policies of interest.
                Defaults to None.

        Returns:
            np.ndarray: (n,2) array of confidence intervals.
        """
        indices = self.indices if cols is None else self.model.get_indices(cols)
        select = [np.where(self.indices == index)[0][0] for index in indices]
        c_alpha = np.quantile(abs(self.projection_rvs).max(axis=1), 1 - alpha)
        return np.array(
            [
                self.params - c_alpha * self.std_params_diag,
                self.params + c_alpha * self.std_params_diag,
            ]
        ).T[select]

    def _make_summary_header(self, alpha: float) -> List[str]:
        return [
            "coef (conventional)",
            "pvalue",
            f"{1-alpha} CI lower",
            f"{1-alpha} CI upper",
        ]


class RQUResults(ResultsBase):
    """Ranked quantile-unbiased results.

    Inherits from :class:`conditional_inference.base.ResultsBase`.

    Args:
        model (RQU): The RQU model instance
        cols (ColumnsType, optional): Names or indices of policies of interest.
            Defaults to None.
        beta (float, optional): Projection quantile for hybrid estimation. Defaults to
            0.
        title (str, optional): Results title. Defaults to "Quantile-unbiased
            estimates".

    Attributes:
        model (RQU): The model instance.
        indices (List[int]): Indices of the policies of interest.
        params (np.ndarray): (n,) array of conventional point estimates.
        pvalues (np.ndarray): (n,) array of probabilities that the true effect of a
            policy is less than 0.
        distributions (List[quantile_unbiased]): Quantile-unbiased distributions
            conditional on rank ordering.
        beta (float): Projection quantile for hybrid estimation.

    Examples:

        .. code-block:: python

            >>> from conditional_inference.rqu import RQU
            >>> import numpy as np
            >>> npolicies = 5
            >>> mean = np.random.normal(size=npolicies)
            >>> cov = np.identity(npolicies)
            >>> rqu = RQU(mean, cov)
            >>> results = rqu.fit(cols="sorted", beta=.005)
            >>> print(results.summary())
                 Quantile-unbiased estimates
            =====================================
               coef (median) pvalue [0.025 0.975]
            -------------------------------------
            x1         0.388  0.412 -2.209  2.931
            x2         0.487  0.413 -2.885  3.672
            x4        -1.289  0.700 -4.354  2.590
            x0        -1.468  0.664 -4.775  2.690
            x3        -0.154  0.529 -3.316  2.183
            ===============
            Dep. Variable y
            ---------------
    """

    def __init__(
        self,
        model: RQU,
        cols: ColumnsType = None,
        beta: float = 0,
        title: str = "Quantile-unbiased estimates",
        **kwargs: Any,
    ):
        super().__init__(model, cols, title)
        self.distributions = self.model.get_distributions(cols, beta=beta, **kwargs)
        self.params = np.array([dist.ppf(0.5) for dist in self.distributions])
        self.pvalues = np.array(
            [(1 - beta) * dist.cdf(0) + beta for dist in self.distributions]
        )
        self.beta = 0 if beta is None else beta

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
        # min-max scale significance level given beta-quantile projection interval
        # see paper for details
        alpha = (alpha - self.beta) / (1 - self.beta)
        return super().conf_int(alpha, cols)

    def _make_summary_header(self, alpha: float) -> List[str]:
        return ["coef (median)", "pvalue", f"[{alpha/2}", f"{1-alpha/2}]"]

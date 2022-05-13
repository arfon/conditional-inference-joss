"""Inference after ranking.

References:

    .. code-block::

        @techreport{andrews2019inference,
            title={ Inference on winners },
            author={ Andrews, Isaiah and Kitagawa, Toru and McCloskey, Adam },
            year={ 2019 },
            institution={ National Bureau of Economic Research }
        }

        @article{andrewsinference,
            title={Inference for Losers},
            author={Andrews, Isaiah and Bowen, Dillon and Kitagawa, Toru and McCloskey, Adam}
        }
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .base import (
    ModelBase,
    ResultsBase,
    ColumnType,
    Numeric1DArray,
)
from .confidence_set import ConfidenceSet
from .stats import quantile_unbiased


class RankConditionResults(ResultsBase):
    """Quantile-unbiased results.

    Inherits from :class:`conditional_inference.base.ResultsBase`.

    Args:
        *args (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
        beta (float, optional): Used to compute the projection quantile for hybrid
            estimation. Defaults to 0.
        marginal_distribution_kwargs (Mapping[str, Any], optional): Passed to
            :meth:`RankCondition.get_marginal_distribution`. Defaults to None.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
    """

    _default_title = "Rank condition quantile-unbiased estimates"

    def __init__(
        self,
        *args: Any,
        beta: float = 0,
        marginal_distribution_kwargs: Mapping[str, Any] = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if marginal_distribution_kwargs is None:
            marginal_distribution_kwargs = {}
        self.marginal_distributions, self.params, self.pvalues = [], [], []
        for i in range(self.model.n_params):
            dist = self.model.get_marginal_distribution(
                i, beta=beta, **marginal_distribution_kwargs
            )
            self.marginal_distributions.append(dist)
            self.params.append(dist.ppf(0.5))
            self.pvalues.append((1 - beta) * dist.cdf(0) + beta)
        self.params, self.pvalues = np.array(self.params), np.array(self.pvalues)
        self._beta = beta

    def _conf_int(self, alpha: float, indices: np.ndarray) -> np.ndarray:
        # see paper for details on adjusting alpha
        return super()._conf_int((alpha - self._beta) / (1 - self._beta), indices)

    def _make_summary_header(self, alpha: float) -> list[str]:
        return ["coef (median)", "pvalue (1-sided)", f"[{alpha/2}", f"{1-alpha/2}]"]


class RankCondition(ModelBase):
    """Rank condition quantile-unbiased estimator.

    Provides utilities for obtaining quantile-unbiased estimates conditional on the
    rank-ordering of conventional estimates of policy effects.

    Subclasses :class:`conditional_inference.base.ModelBase`.

    Args:
        *args (Any): Passed to :class:`conditional_inference.base.ModelBase`.
        xmean (Numeric1DArray, optional): (# params,) array of conventional estimates to
            use for ranking. If None, ranking conditions are based on ``mean``. Defaults
            to None.
        xycov (np.ndarray, optional): (# params, # params) covariance matrix between
            ``mean`` and ``xmean``. Defaults to None.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ModelBase`.

    Raises:
        ValueError: Either all or none of ``xmean``, ``xcov`` and ``xycov`` must be
            specified.

    Additional attributes:
        xmean (np.ndarray): (# params,) array of conventional estimates to use for
            ranking.
        xycov (np.ndarray): (# params, # params) covariance matrix between ``self.mean``
            and ``self.xmean``.

    Examples:

        Compute a quantile-unbiased distribution of the x4 parameter given that it was
        the top-ranked parameter.

        .. testcode::

            import numpy as np
            from conditional_inference.rank_condition import RankCondition

            model = RankCondition(np.arange(5), np.identity(5))
            dist = model.get_marginal_distribution("x4")
            print(dist.ppf([.025, .5, .975]))

        .. testoutput::

            [0.06742731 3.68627552 5.93267239]

        Compute an "almost" quantile-unbiased hybrid distribution.

        .. testcode::

            dist = model.get_marginal_distribution("x4", beta=.005)
            print(dist.ppf([.025, .5, .975]))

        .. testoutput::

            [0.8674603  3.68732401 5.9328792 ]

        Summarize the quantile-unbiased results.

        .. testcode::

            results = model.fit()
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

            Rank condition quantile-unbiased estimates
            ===============================================
            coef (median) pvalue (1-sided) [0.025 0.975]
            -----------------------------------------------
            x0         0.314            0.406 -1.933  3.933
            x1         1.000            0.285 -2.922  4.922
            x2         2.000            0.136 -1.922  5.922
            x3         3.000            0.058 -0.922  6.922
            x4         3.686            0.023  0.067  5.933
            ===============
            Dep. Variable y
            ---------------

        Summarize the "almost" quantile-unbiased hybrid results.

        .. testcode::

            results = model.fit(beta=.005)
            print(results.summary())

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

            Rank condition quantile-unbiased estimates
            ===============================================
            coef (median) pvalue (1-sided) [0.025 0.975]
            -----------------------------------------------
            x0         0.313            0.409 -1.977  3.152
            x1         1.000            0.288 -2.152  4.152
            x2         2.000            0.140 -1.152  5.152
            x3         3.000            0.044 -0.152  6.152
            x4         3.687            0.005  0.848  5.977
            ===============
            Dep. Variable y
            ---------------
    """

    _results_cls = RankConditionResults

    def __init__(
        self,
        *args: Any,
        xmean: Numeric1DArray = None,
        xycov: np.ndarray = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        xparams = (xmean, xycov)
        if any([x is not None for x in xparams]) and None in xparams:
            raise ValueError(
                "Either both or neither of `xmean` and `xycov` must be given."
            )

        self.xmean = xmean
        self.xycov = xycov
        self._confidence_set = ConfidenceSet(self.mean, self.cov).fit()

    @property
    def xmean(self):  # pylint: disable=missing-function-docstring
        return self.mean if self._xmean is None else self._xmean

    @xmean.setter
    def xmean(self, xmean):  # pylint: disable=missing-function-docstring
        self._xmean = None if xmean is None else np.array(xmean)
        self._estimated_ranks = (-self.xmean).argsort().argsort()

    @property
    def xycov(self):  # pylint: disable=missing-function-docstring
        return self.cov if self._xycov is None else self._xycov

    @xycov.setter
    def xycov(self, xycov):  # pylint: disable=missing-function-docstring
        self._xycov = None if xycov is None else np.array(xycov)

    def get_marginal_distribution(  # pylint: disable=too-many-arguments
        self,
        column: ColumnType,
        ranks: Numeric1DArray = None,
        beta: float = 0,
        **kwargs: Any,
    ) -> quantile_unbiased:
        """Compute a quantile-unbiased distribution for a given ranking condition.

        Args:
            column (ColumnType): Name or index of the parameter of interest. Defaults to
                None.
            ranks (Numeric1DArray, optional): Ranking conditions for the parameter of
                interest. This method returns a quantile-unbiased distribution given
                that the estimated rank of the parameter of interest is in ``ranks``.
                If None, the estimated rank of the parameter is used as the ranking
                condition. Defaults to None.
            beta (float, optional): Used to compute the projection quantile for hybrid
                estimation. Defaults to 0.
            **kwargs (Any): Passed to :class:`quantile_unbiased`.

        Returns:
            quantile_unbiased: Quantile-unbiased distribution.
        """

        def rank_condition_holds():
            return (
                (current_rank - n_always_equal <= ranks)
                & (ranks <= current_rank + n_always_equal)
            ).any()

        index = self.get_index(column)
        ranks = np.atleast_1d(self._estimated_ranks[index] if ranks is None else ranks)

        # compute the conditional truncation set
        equal_cov = self.xycov[index, index] == self.xycov[:, index]
        greater_cov = self.xycov[~equal_cov, index] > self.xycov[index, index]
        # number of parameters whose estimates are always greater than the estimate of
        # the target parameter
        n_always_greater = (equal_cov & (self.xmean > self.xmean[index])).sum()
        # number of parameters whose estimates are always equal to the estimate of the
        # target parameter. -1 so you don't count the target parameter itself.
        n_always_equal = (equal_cov & (self.xmean == self.xmean[index])).sum() - 1
        z = (  # pylint: disable=invalid-name
            self.xmean
            - (self.xycov[:, index] / self.cov[index, index]) * self.mean[index]
        )
        # thresholds are the values at which the estimated value of the target parameter
        # equals the estimated value of the other parameters (denoted Q in the paper)
        thresholds = (
            self.cov[index, index]
            * (z[~equal_cov] - z[index])
            / (self.xycov[index, index] - self.xycov[~equal_cov, index])
        )
        argsort = thresholds.argsort()
        greater_cov, thresholds = greater_cov[argsort], thresholds[argsort]
        intervals = np.array([thresholds, np.append(thresholds[1:], np.inf)]).T
        current_rank = n_always_greater + (~greater_cov).sum()
        truncset = [[-np.inf, thresholds[0]]] if rank_condition_holds() else []
        for greater_cov_i, interval in zip(greater_cov, intervals):
            current_rank += 1 if greater_cov_i else -1
            if rank_condition_holds():
                truncset.append(interval)

        if beta != 0:
            # projection interval must be centered on 0 for `quantile_unbiased`
            kwargs["projection_interval"] = (
                self._confidence_set.conf_int(beta, [index]) - self.mean[index]
            )[0]
        return quantile_unbiased(  # type: ignore
            y=self.mean[index],
            scale=np.sqrt(self.cov[index, index]),
            truncation_set=truncset,
            **kwargs,
        )

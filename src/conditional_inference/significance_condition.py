"""Inference for parameters that achieve statistical significance.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from conditional_inference.base import ColumnType, ModelBase, ResultsBase
from conditional_inference.confidence_set import ConfidenceSet
from conditional_inference.stats import quantile_unbiased


class SignificanceConditionResults(ResultsBase):
    """Quantile-unbiased results.

    Sublcasses :class:`conditional_inference.base.ResultsBase`.

    Args:
        *args (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
        marginal_distribution_kwargs (Mapping[str, Any], optional): Passed to
            :meth:`SignificanceCondition.get_marginal_distribution`. Defaults to None.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ResultsBase`.
    """

    _default_title = "Significance condition quantile-unbiased estimates"

    def __init__(
        self,
        *args: Any,
        marginal_distribution_kwargs: Mapping[str, Any] = None,
        **kwargs: Any,
    ):

        super().__init__(*args, **kwargs)
        if marginal_distribution_kwargs is None:
            marginal_distribution_kwargs = {}
        self.marginal_distributions, self.params, self.pvalues = [], [], []
        for i in range(self.model.n_params):
            dist = self.model.get_marginal_distribution(
                i, **marginal_distribution_kwargs
            )
            self.marginal_distributions.append(dist)
            self.params.append(dist.ppf(0.5))
            self.pvalues.append(dist.cdf(0))
        self.params, self.pvalues = np.array(self.params), np.array(self.pvalues)

    def _make_summary_header(self, alpha: float) -> list[str]:
        return ["coef (median)", "pvalue (1-sided)", f"[{alpha/2}", f"{1-alpha/2}]"]


class SignificanceCondition(ModelBase):
    """Significance condition quantile-unbiased estimator.

    Subclasses :class:`conditional_inference.base.ModelBase`.

    Examples:
        Get a quantile-unbiased distribution for x3.

        .. testcode::

            import numpy as np
            from conditional_inference.significance_condition import SignificanceCondition

            model = SignificanceCondition(np.arange(4), np.identity(4))
            dist = model.get_marginal_distribution("x3")
            print(dist.ppf([.025, .5, .975]))

        .. testoutput::

            [-0.32622156  1.95349752  4.8138274 ]

        Display the results.

        .. testcode::

            results = model.fit()
            print(results.summary(columns=["x3"]))

        .. testoutput::
            :options: -ELLIPSIS, +NORMALIZE_WHITESPACE

            Significance condition quantile-unbiased estimates
            ===============================================
               coef (median) pvalue (1-sided) [0.025 0.975]
            -----------------------------------------------
            x3         1.953            0.106 -0.326  4.814
            ===============
            Dep. Variable y
            ---------------
    """

    _results_cls = SignificanceConditionResults

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._confidence_set = ConfidenceSet(self.mean, self.cov).fit()

    def get_marginal_distribution(
        self, column: ColumnType, alpha: float = 0.05, **kwargs: Any
    ) -> quantile_unbiased:
        """Get the marginal quantile-unbiased distribution.

        The distribution is quantile-unbiased conditional on the parameter being
        statistically significant at level ``alpha``.

        Args:
            column (ColumnType): Selected column.
            alpha (float, optional): Significance level. Defaults to .05.

        Returns:
            quantile_unbiased: Quantile-unbiased distribution.
        """
        index = self.get_index(column)
        critical_value = (
            self._confidence_set.conf_int(alpha, [index]) - self.mean[index]
        )[0, 1]
        truncation_set = [[-np.inf, -critical_value], [critical_value, np.inf]]
        return quantile_unbiased(
            y=self.mean[index],
            scale=np.sqrt(self.cov[index, index]),
            truncation_set=truncation_set,
            **kwargs,
        )

"""Quantile-unbiased effect size estimates for policies that achieve statistical significance.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from conditional_inference.base import ColumnsType, ModelBase, ResultsBase
from conditional_inference.stats import quantile_unbiased
from conditional_inference.utils import compute_projection_quantile


class SQU(ModelBase):
    """Statistical significance quantile-unbiased estimator.

    Inherits from :class:`conditional_inference.base.ModelBase`.

    Args:
        *args (Any): Passed to :class:`conditional_inference.base.ModelBase`.
        seed (int, optional): Random seed. Defaults to 0.
        **kwargs (Any): Passed to :class:`conditional_inference.base.ModelBase`.

    Attributes:
        seed (int): Random seed.
    """
    def __init__(self, *args: Any, seed: int=0, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def fit(self, cols: ColumnsType=None, alpha: float=0.05, two_sided:bool=True, **kwargs: Any) -> SQUResults:
        """Fit the SQU estimator.

        Args:
            cols (ColumnsType, optional): Names or indices of the policies of interest.
                Defaults to None.
            alpha (float, optional): Family-wise error rate. Defaults to 0.05.
            two_sided (bool, optional): Indicates that hypothesis tests should be
                two-sided (as opposed to one-sided). Defaults to True.

        Returns:
            SQUResults: Results.
        """
        return SQUResults(self, cols, alpha=alpha, two_sided=two_sided, **kwargs)

    def compute_critical_values(self, alpha: float=0.05, two_sided: bool=True, n_samples: int=10000) -> np.ndarray:
        """Compute values at which each hypothesis would be rejected.

        Args:
            alpha (float, optional): Family-wise error rate. Defaults to 0.05.
            two_sided (bool, optional): Indicates that hypothesis tests should be
                two-sided (as opposed to one-sided). Defaults to True.
            n_samples (int, optional): Number of samples used to approximate the
                critical value. Defaults to 10000.

        Returns:
            np.ndarray: (# policies,) array of critical values.
        """
        return (
            compute_projection_quantile(
                self.mean,
                self.cov,
                (alpha / 2) if two_sided else alpha,
                n_samples=n_samples,
                random_state=self.seed,
            )
            * np.sqrt(self.cov.diagonal())
        )

    def get_distributions(self, cols:ColumnsType=None, alpha:float=0.05, two_sided:bool=True, n_samples: int=10000) -> list[quantile_unbiased]:
        """Get quantile-unbiased distributions.

        Args:
            cols (ColumnsType, optional): Names or indices of the policies of interest.
                Defaults to None.
            alpha (float, optional): Family-wise error rate. Defaults to 0.05.
            two_sided (bool, optional): Indicates the hypothesis tests should be
                two-sided (as opposed to one-sided). Defaults to True.
            n_samples (int, optional): Number of samples used to approximate critical
                values. Defaults to 10000.

        Returns:
            list[quantile_unbiased]: Quantile-unbiased distributions for selected
                policies.
        """
        def get_truncation_set(critical_value):
            if two_sided:
                return [(-np.inf, -critical_value), (critical_value, np.inf)]
            return [(critical_value, np.inf)]

        indices = self.get_indices(cols)
        critical_values = self.compute_critical_values(
            alpha, two_sided, n_samples=n_samples
        )
        return [
            quantile_unbiased(
                y=self.mean[i],
                scale=np.sqrt(self.cov[i, i]),
                truncation_set=get_truncation_set(critical_values[i]),
            )
            for i in indices
        ]


class SQUResults(ResultsBase):
    """Significance quantile-unbiased results.

    Args:
        model (SQU): The model instance.
        cols (ColumnsType, optional): Names or indices of policies of interest. Defaults
            to None.
        alpha (float, optional): Family-wise error rate. Defaults to 0.05.
        two_sided (bool, optional): Indicates hypothesis tests should be two-sided (as
            opposed to one-sided). Defaults to True.
        n_samples (int, optional): Number of samples used to approximate critical
            values. Defaults to 10000.
        title (str, optional): Results title. Defaults to "Quantile-unbiased estimates".

    Attributes:
        model (SQU): The model instance.
        indices (list[int]): Indices of the policies of interest.
        distributions (list[quantile_unbiased]): Quantile-unbiased distributions for
            selected policies.
        params (np.ndarray): (# policies,) array of median-unbiased point estimates.
        pvalues (np.ndarray): (# policies,) array of probabilities that the true effect
            of a policy is less than 0.

    Raises:
        RuntimeError: If no policies achieved statistical significance.
    """
    def __init__(
        self,
        model: SQU,
        cols: ColumnsType=None,
        alpha: float=0.05,
        two_sided: bool=True,
        n_samples: int=10000,
        title: str="Quantile-unbiased estimates",
    ):
        critical_values = model.compute_critical_values(alpha, two_sided, n_samples)
        if two_sided:
            significant_indices = critical_values < abs(model.mean)
        else:
            significant_indices = critical_values < model.mean
        indices = np.where(significant_indices)[0]
        if cols is not None:
            indices = list(
                set(list(model.get_indices(cols)) + list(significant_indices))
            )

        if len(indices) == 0:
            raise RuntimeError("No policies were statistically significant.")

        super().__init__(model, indices, title)
        self.distributions = model.get_distributions(
            indices, alpha, two_sided, n_samples
        )
        self.params = np.array([dist.ppf(0.5) for dist in self.distributions])
        self.pvalues = np.array([dist.cdf(0) for dist in self.distributions])

    def _make_summary_header(self, alpha: float):
        return ["coef (median)", "pvalue", f"[{alpha/2}", f"{1-alpha/2}]"]

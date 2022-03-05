from __future__ import annotations

import numpy as np
from conditional_inference.base import ModelBase, ResultsBase
from conditional_inference.stats import quantile_unbiased
from conditional_inference.utils import compute_projection_quantile


class SQU(ModelBase):
    def __init__(self, *args, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed

    def fit(self, cols=None, alpha=0.05, two_sided=True, **kwargs):
        return SQUResults(self, cols, alpha=alpha, two_sided=two_sided, **kwargs)

    def compute_critical_values(self, alpha=0.05, two_sided=True, n_samples=10000):
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

    def get_distributions(self, cols=None, alpha=0.05, two_sided=True, n_samples=10000):
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
    def __init__(
        self,
        model,
        cols=None,
        alpha=0.05,
        two_sided=True,
        n_samples=10000,
        title="Quantile-unbiased estimates",
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
            raise RuntimeError("No parameters were statistically significant.")

        super().__init__(model, indices, title)
        self.distributions = model.get_distributions(
            indices, alpha, two_sided, n_samples
        )
        self.params = np.array([dist.ppf(0.5) for dist in self.distributions])
        self.pvalues = np.array([dist.cdf(0) for dist in self.distributions])

    def _make_summary_header(self, alpha: float):
        return ["coef (median)", "pvalue", f"[{alpha/2}", f"{1-alpha/2}]"]

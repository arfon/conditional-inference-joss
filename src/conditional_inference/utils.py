"""Conditional inference utilities
"""
from typing import Any, Optional, Sequence, Union

import numpy as np
from scipy.stats import multivariate_normal, wasserstein_distance

Numeric1DArray = Sequence[float]


def _get_sample_weight(sample_weight: Optional[np.ndarray], shape: int) -> np.ndarray:
    if sample_weight is None:
        sample_weight = np.ones(shape)
    sample_weight = np.array(sample_weight)
    return sample_weight / sample_weight.sum()


def expected_wasserstein_distance(
    mean: Numeric1DArray,
    cov: np.ndarray,
    estimated_means: np.ndarray,
    sample_weight: np.ndarray = None,
    **kwargs: Any
) -> float:
    """Compute the expected Wasserstein distance.

    This loss function computes the Wasserstein distance between the observed means
    ``mean`` and the distribution of means you would expect to observe given the
    estimated population means ``estimated_means``.

    Args:
        mean (Numeric1DArray): (n,) array of observed sample means.
        cov (np.ndarray): (n, n) covariance matrix of sample means.
        estimated_means (np.ndarray): (# samples, n) matrix of draws from
            a distribution of population means.
        sample_weight (np.ndarray, optional): (# samples,) array of sample weights for
            ``estimated_means``. Defaults to None.
        **kwargs (Any): Keyword arguments for ``scipy.stats.wasserstein_distance``.

    Returns:
        float: Loss.
    """

    def compute_distance(estimated_mean):
        dist = multivariate_normal(estimated_mean, cov)
        return wasserstein_distance(dist.rvs(), mean, **kwargs)

    sample_weight = _get_sample_weight(sample_weight, estimated_means.shape[0])
    distances = np.apply_along_axis(compute_distance, 1, estimated_means)
    return (sample_weight * distances).sum()


def weighted_quantile(
    values: np.ndarray,
    quantiles: Union[float, Numeric1DArray],
    sample_weight: np.ndarray = None,
    values_sorted: bool = False,
) -> np.ndarray:
    """Compute weighted quantiles.

    Args:
        values (np.ndarray): (n,) array over which to compute quantiles.
        quantiles (Union[float, Numeric1DArray]): (k,) array of quantiles of interest.
        sample_weight (np.ndarray, optional): (n,) array of sample weights. Defaults to
            None.
        values_sorted (bool, optional): Indicates that ``values`` have been pre-sorted.
            Defaults to False.

    Returns:
        np.array: (k,) array of weighted quantiles.

    Acknowledgements:
        Credit to `Stackoverflow <https://stackoverflow.com/a/29677616/10676300>`_.
    """
    values = np.array(values)
    quantiles = np.atleast_1d(quantiles)  # type: ignore
    sample_weight = _get_sample_weight(sample_weight, len(values))
    assert np.all(quantiles >= 0) and np.all(  # type: ignore
        quantiles <= 1  # type: ignore
    ), "quantiles should be in [0, 1]"

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight  # type: ignore
    return np.interp(quantiles, weighted_quantiles, values)


def weighted_cdf(
    values: np.ndarray, x: float, sample_weight: np.ndarray = None
) -> float:
    """Compute weighted CDF.

    Args:
        values (np.ndarray): (n,) array over which to compute the CDF.
        x (float): Point at which to evaluate the CDF.
        sample_weight (np.ndarray, optional): (n,) array of sample weights. Defaults to
            None.

    Returns:
        float: CDF of ``values`` evaluated at ``x``.
    """
    sample_weight = _get_sample_weight(sample_weight, len(values))
    return (sample_weight * (np.array(values) < x)).sum()

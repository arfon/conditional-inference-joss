"""Conditional inference utilities
"""
from multiprocessing.sharedctypes import Value
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, wasserstein_distance
from statsmodels.base.model import LikelihoodModelResults

Numeric1DArray = Sequence[float]


def _get_sample_weight(sample_weight: Optional[np.ndarray], shape: int) -> np.ndarray:
    if sample_weight is None:
        sample_weight = np.ones(shape)
    sample_weight = np.array(sample_weight)
    return sample_weight / sample_weight.sum()


def compute_projection_rvs(
    mean: np.array, cov: np.array, size: int = 1, random_state: int = None
) -> np.ndarray:
    """Sample random values to construct projection confidence intervals.

    Args:
        mean (np.array): (# policies,) array of means.
        cov (np.array): (# policies, # policies) covariance matrix.
        size (int, optional): Number of samples. Defaults to 1.
        random_state (int, optional): Random state passed to
            ``scipy.stats.multivariate_normal``. Defaults to None.

    Returns:
        np.ndarray: (size, 2) array of samples.
    """
    rvs = multivariate_normal.rvs(
        np.zeros(len(mean)), cov, size=size, random_state=random_state
    )
    if len(rvs.shape) == 1:
        rvs = np.atleast_2d(rvs).T
    rvs /= np.sqrt(cov.diagonal())
    return np.array([rvs.min(axis=1), rvs.max(axis=1)]).T


def compute_projection_quantile(
    mean: np.array,
    cov: np.array,
    alpha: float = 0.05,
    n_samples: int = 10000,
    random_state: int = None,
) -> float:
    """Compute the 1-alpha quantile for projection confidence intervals.

    Args:
        mean (np.array): (# policies,) array of means.
        cov (np.array): (# policies, # policies) covariance matrix.
        alpha (float, optional): Quantile level of the projection CI. Defaults to 0.05.
        n_samples (int, optional): Number of samples used in approximating the 1-alpha
            quantile. Defaults to 10000.
        random_state (int, optional): Random state passed to
            ``scipy.stats.multivariate_normal``. Defaults to None.

    Returns:
        float: 1-alpha quantile of the projection CI.
    """
    if alpha == 0:
        return np.inf
    if len(mean) == 1:
        return norm.ppf(1 - alpha, 0, np.sqrt(cov))[0]
    rvs = compute_projection_rvs(mean, cov, size=n_samples, random_state=random_state)
    return np.quantile(abs(rvs).max(axis=1), 1 - alpha)


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


def holm_bonferroni_correction(
    filename: str = None, results: LikelihoodModelResults = None, alpha: float = 0.05
) -> pd.Series:
    """Get significant coefficients by performing a Holm-Bonferroni correction.

    Args:
        filename (str, optional): Name of the csv file with conventional estimates.
            Defaults to None.
        results (LikelihoodModelResults, optional): Results. Defaults to None.
        alpha (float, optional): Significance level. Defaults to .05.

    Raises:
        ValueError: You must specify either ``filename`` or ``results`` but not both.

    Returns:
        pd.DataFrame: Dataframe indicating which coefficients are significant.

    Notes:
        If you input a ``filename``, this correction looks at one-tailed hypothesis
        tests.
    """
    if filename is None and results is None:
        raise ValueError("filename or results must be specified.")

    if filename is not None and results is not None:
        raise ValueError("Please specify either filename or results; not both.")

    if results is None:
        from .bayes.classic import LinearClassicBayes

        results = LinearClassicBayes.from_csv(filename, prior_cov=np.inf).fit()

    argsort = results.pvalues.argsort()
    df = pd.DataFrame(
        {"pvalues": results.pvalues[argsort]},
        index=np.array(results.model.exog_names)[argsort],
    )
    index = np.where(df.pvalues > alpha / (len(df) - np.arange(len(df))))[0][0]
    df["significant"] = np.arange(len(df)) < index
    return df


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

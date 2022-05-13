"""Statistical distributions.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, List, Sequence, Tuple, Union

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.misc import derivative
from scipy.optimize import NonlinearConstraint, fsolve, minimize
from scipy.stats import norm, rv_continuous, truncnorm as truncnorm_base


from .base import Numeric1DArray
from .utils import weighted_quantile


class joint_distribution:
    """Join distribution based on independent marginal distributions.

    Args:
        marginal_distributions (Sequence[rv_continuous]): Marginal distributions.
    """

    def __init__(self, marginal_distributions: Sequence[rv_continuous]):
        self._marginal_distributions = list(marginal_distributions)

    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of the probability density function evaluated at ``x``.

        Args:
            x (np.ndarray): (n, # marginals) matrix of values at which to evaluate the
                density function.

        Returns:
            np.ndarray: (n,) array of log density.
        """
        x = np.array(x).reshape(-1, len(self._marginal_distributions))
        return np.sum([dist.logpdf(x_i) for dist, x_i in zip(self._marginal_distributions, x.T)], axis=0)

    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function evaluated at ``x``.

        Args:
            x (np.ndarray): (n, # marginals) matrix of values at which to evaluate the
                density function.

        Returns:
            np.ndarray: (n,) array of densities.
        """
        return np.exp(self.logpdf(x))

    def rvs(self, size: int = 1) -> np.ndarray:
        """Sample random values.

        Args:
            size (int, optional): Number of samples to draw. Defaults to 1.

        Returns:
            np.ndarray: (size, # marginals) matrix of samples.
        """
        return np.vstack(
            [dist.rvs(size=size) for dist in self._marginal_distributions]
        ).T


class mixture(rv_continuous):
    """Mixture distribution.

    Args:
        distributions (list[rv_continuous]): List of n distributions to mix over.
        weights (Numeric1DArray, optional): (n,) array of mixture weights. Defaults to None.

    Attributes:
        distributions (list[rv_continuous]): Distributions to mix over.
        weights (np.ndarray): Mixture weights.
    """

    def __init__(
        self,
        distributions: list[rv_continuous],
        weights: Numeric1DArray = None,
        **kwargs: Any,
    ):

        super().__init__(**kwargs)
        self.distributions = distributions
        self.weights = (
            np.ones(len(distributions)) if weights is None else np.atleast_1d(weights)
        )
        self.weights /= self.weights.sum()

    def _pdf(self, x):
        return (
            self.weights * np.array([dist.pdf(x) for dist in self.distributions]).T
        ).sum(axis=1)

    def _cdf(self, x):
        return (
            self.weights * np.array([dist.cdf(x) for dist in self.distributions]).T
        ).sum(axis=1)

    def mean(self):
        return (
            self.weights * np.array([dist.mean() for dist in self.distributions])
        ).sum()

    def var(self):
        return (
            self.weights * np.array([dist.var() for dist in self.distributions])
        ).sum()

    def std(self):
        return np.sqrt(self.var())


class nonparametric(rv_continuous):
    """Nonparametric distribution.

    Args:
        values (tuple[np.array, np.array]): (n,) array of x values, (n,) array of the
            probability mass function evaluated at x.
        kind (str, optional): Type of interpolation to use. Passed to
            ``scipy.interpolate.interp1d``. Defaults to None.

    Attributes:
        xk (np.ndarray): (n,) array of x values.
        pk (np.ndarray): (n,) array of the probability mass function evaluated at x.

    Notes:
        This distribution interpolates between the probability mass function to
        "continuize" the discrete function.
    """

    def __init__(self, values, kind=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.xk, self.pk = np.array(values[0], float), np.array(values[1], float)
        self.pk /= self.pk.sum()
        self._cdf_values = np.cumsum(self.pk)
        if kind is not None:
            self._kind = kind
        else:
            self._kind = "cubic" if len(self.xk) > 3 else "linear"
        self._scale = 1
        self._scale = 1 / quad(self._pdf, self.xk[0], self.xk[-1])[0]

    def _pdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        pdf = np.zeros(len(x))
        in_range = (self.xk[0] < x) & (x < self.xk[-1])
        pdf[in_range] = interp1d(self.xk, self.pk, kind=self._kind)(x[in_range])
        return self._scale * pdf

    def _cdf(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x)
        cdf = np.zeros(len(x))
        cdf[x >= self.xk[-1]] = 1
        in_range = (self.xk[0] < x) & (x < self.xk[-1])
        cdf[in_range] = interp1d(self.xk, self._cdf_values, kind=self._kind)(
            x[in_range]
        )
        return cdf

    def _ppf(self, q: np.ndarray) -> np.ndarray:
        return weighted_quantile(self.xk, q, self.pk)

    def moment(self, func: Callable[[np.ndarray], np.ndarray]) -> float:
        """Compute a moment.

        Args:
            func (Callable[[np.ndarray], np.ndarray]): Moment function that takes
                ``self.xk`` and returns an array of the same shape.

        Returns:
            float: Moment.
        """
        return sum(self.pk * func(self.xk))

    def mean(self) -> float:
        """Compute the mean.

        Returns:
            float: Mean.
        """
        return self.moment(lambda x: x)

    def var(self) -> float:
        """Compute the variance.

        Returns:
            float: Variance.
        """
        mean = self.mean()
        return self.moment(lambda x: (x - mean) ** 2)

    def std(self) -> float:
        """Compute the standard deviation.

        Returns:
            float: Standard deviation.
        """
        return np.sqrt(self.var())


class quantile_unbiased(rv_continuous):  # pylint: disable=invalid-name
    """Conditional quantile-unbiased distribution.

    Inherits from `scipy.stats.rv_continuous <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html>`_
    and handles standard public methods (``pdf``, ``cdf``, etc.).

    Args:
        y (float): Value at which the truncated CDF is evaluated
        projection_interval (Union[float, Tuple[float, float]], optional): Lower and
            upper bounds of the projection confidence interval. Defaults to
            (-np.inf, np.inf).
        bounds (Tuple[float, float], optional): Lower and upper bounds of the support
            of the distribution. Defaults to (-np.inf, np.inf).
        dx (float): Used to numerically approximate the PDF.
        **truncnorm_kwargs (Any): Keyword arguments for :class:`truncnorm`.

    Attributes:
        y (float): Value at which the truncated CDF is evaluated.
        bounds (Tuple[float, float]): Lower and upper bound of the support of the
            distribution.
        dx (float): Used to numerically approximate the PDF.
        truncnorm_kwargs (dict): Keyword arguments for :class:`truncnorm`.

    Examples:
        Compute a median-unbiased estimate of a normally distributed variable given
            that its observed value is 1 and falls between 0 and 3.

        .. doctest::

            >>> from conditional_inference.stats import quantile_unbiased
            >>> dist = quantile_unbiased(1, truncation_set=[(0, 3)])
            >>> dist.ppf(.5)
            0.7108033900602351
    """

    def __init__(
        self,
        y: float,
        projection_interval: Union[float, Tuple[float, float]] = (-np.inf, np.inf),
        bounds: Tuple[float, float] = (-np.inf, np.inf),
        dx: float = None,
        **truncnorm_kwargs: Any,
    ):
        super().__init__()
        self.y = y  # pylint: disable=invalid-name
        if np.isscalar(projection_interval):
            projection_interval = abs(projection_interval)  # type: ignore
            projection_interval = (-projection_interval, projection_interval)
        self.projection_interval = tuple(projection_interval)
        self.bounds = bounds
        self.truncnorm_kwargs = truncnorm_kwargs
        self.dx = dx

        self._cdf_min: float = (
            0  # type: ignore
            if bounds[0] == -np.inf
            else 1 - self._truncated_cdf(np.array([bounds[0]]))
        )
        if self._cdf_min >= 1:
            warnings.warn(
                "Untruncated CDF of lower bound == 1: try decreasing the lower bound",
                RuntimeWarning,
            )
        self._cdf_max: float = (
            1  # type: ignore
            if bounds[1] == np.inf
            else 1 - self._truncated_cdf(np.array([bounds[1]]))
        )
        if self._cdf_max <= 0:
            warnings.warn(
                "Untruncated CDF of upper bound == 0: try increasing the upper bound",
                RuntimeWarning,
            )

    @property
    def dx(self):  # pylint: disable=missing-docstring
        # create a default dx value if this property has not yet been set
        if self._dx is None:
            self.dx = np.diff(self.ppf([0.95, 0.05])) / 50
        return self._dx

    @dx.setter
    def dx(self, value):
        self._dx = value

    @property
    def bounds(self):  # pylint: disable=missing-docstring
        # Potentially restrict the bounds of this distribution to ensure the truncation
        # set and # projection interval overlap
        truncation_set = self.truncnorm_kwargs.get("truncation_set")
        if truncation_set is None or self.projection_interval == (-np.inf, np.inf):
            return self._bounds

        a, b = zip(*truncation_set)  # pylint: disable=invalid-name
        return (
            max(self._bounds[0], np.min(a) - self.projection_interval[1]),
            min(self._bounds[1], np.max(b) - self.projection_interval[0]),
        )

    @bounds.setter
    def bounds(self, bounds):
        self._bounds = bounds

    def _truncated_cdf(  # pylint: disable=invalid-name
        self, x: np.ndarray
    ) -> np.ndarray:
        """Compute the truncated CDF evaluated at ``self.y`` with shift ``x``."""

        def truncated_cdf(x_i):
            # get the intersection of the truncation set and the projection confidence
            # interval centered on x_i
            intersection = []
            for interval in truncation_set:
                clipped_interval = (
                    float(max(interval[0], x_i + self.projection_interval[0])),
                    float(min(interval[1], x_i + self.projection_interval[1])),
                )
                if clipped_interval[0] < clipped_interval[1]:
                    interval = (
                        normalize(clipped_interval[0], x_i),
                        normalize(clipped_interval[1], x_i),
                    )
                    intersection.append(interval)

            return truncnorm(intersection, loc=x_i, **truncnorm_kwargs).cdf(self.y)

        def normalize(value, loc):
            return (value - loc) / scale

        truncation_set = self.truncnorm_kwargs.get("truncation_set")
        scale = self.truncnorm_kwargs.get("scale", 1)
        truncnorm_kwargs = {
            key: value
            for key, value in self.truncnorm_kwargs.items()
            if key != "truncation_set"
        }
        rval = np.array([truncated_cdf(x_i) for x_i in x])
        return rval[0] if len(rval) == 1 else rval

    def _cdf(self, x: np.ndarray) -> np.ndarray:  # pylint: disable=arguments-differ
        """Cumulative distribution function.

        Args:
            x (np.ndarray): (n,) array of values at which to evaluate the CDF.

        Returns:
            np.ndarray: (n,) array of evaluations.
        """
        if self._cdf_min >= 1 or self._cdf_max <= 0:

            def handle_cdf_out_of_bounds(x_i):
                return self.bounds[0] <= x_i if self.y < x_i else self.bounds[1] < x_i

            warnings.warn(
                "Untruncated CDF of lower bound >= 1 or CDF of upper bound <= 0"
            )
            return np.array([handle_cdf_out_of_bounds(x_i) for x_i in x]).astype(float)

        cdf = (1 - self._truncated_cdf(x) - self._cdf_min) / (
            self._cdf_max - self._cdf_min
        )
        return ((self.bounds[0] < x) & (x < self.bounds[1])) * cdf + (
            self.bounds[1] <= x
        ).astype(float)

    def _pdf(self, x: np.ndarray) -> np.ndarray:  # pylint: disable=arguments-differ
        """Probability density function.

        Args:
            x (np.ndarray): (n,) array of values at which to evaluate the PDF.

        Returns:
            np.ndarray: (n,) array of evaluations.
        """
        pdf = derivative(self._cdf, x, dx=self.dx, order=5)
        return ((self.bounds[0] < x) & (x < self.bounds[1])) * pdf

    def _ppf(self, q: np.ndarray) -> np.ndarray:  # pylint: disable=arguments-differ
        """See self.ppf."""

        def func(mu, q_i):
            return self._truncated_cdf(mu) - (1 - q_i)

        if self._cdf_min >= 1 or self._cdf_max <= 0:
            warnings.warn(
                "Untruncated CDF of lower bound >= 1 or CDF of upper bound <= 0"
            )

            value = self.bounds[0] if self.bounds[0] > self.y else self.bounds[1]
            return np.full(q.shape, value)

        q_t = np.atleast_1d(q) * (self._cdf_max - self._cdf_min) + self._cdf_min
        return np.array([fsolve(func, [self.y], args=(q_i,))[0] for q_i in q_t])

    def ppf(  # pylint: disable=arguments-differ
        self, q: Union[float, Numeric1DArray]
    ) -> np.ndarray:
        """Percent point function.

        Args:
            q (np.ndarray): (n,) array of quantiles at which to evaluate the PPF.

        Returns:
            np.ndarray: (n,) array of evaluations.
        """
        return np.clip(super().ppf(q), *self.bounds)


class truncnorm(rv_continuous):  # pylint: disable=invalid-name
    """Truncated normal distribution.

    Inherits from `scipy.stats.rv_continuous <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html>`_
    and handles standard public methods (``pdf``, ``cdf``, etc.).

    This uses the [exponential tilting](https://ieeexplore.ieee.org/document/7408180)
    approximation method.

    Args:
        truncation_set (List[Tuple[float, float]], optional): List of truncation
            intervals, e.g., ``[(-1, 0), (1, 2)]`` truncates the distribution to
            [-1, 0] union [1, 2]. Defaults to None.
        loc (float, optional): Location. Defaults to 0.
        scale (float, optional): Scale parameter. Defaults to 1.
        n_samples (int, optional): Number of samples to draw for approximation.
            Defaults to 10000.
        seed (int, optional): Random seed. Defaults to 0.

    Attributes:
        loc (float): Location parameter.
        scale (float): Scale parameter.
        lower_bound (np.array): (# intervals,) array of lower bounds of the truncation
            intervals.
        upper_bound (np.array): (# intervals,) array of upper bounds of the truncation
            intervals.
        interval_masses (np.array): (# intervals,) array of the amount of mass in each
            truncation interval.
        n_samples (int): Number of samples to draw for approximation. Defaults to
            10000.

    Note:
        The truncation set is defined over the domain of the standard normal. To
        convert the truncation set for a specific mean and standard deviation, use:

        .. code-block:: python

            >>> truncation_set = [(myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std)]
    """

    def __init__(
        self,
        truncation_set: List[Tuple[float, float]] = None,
        loc: float = 0,
        scale: float = 1,
        n_samples: int = 10000,
        seed: int = 0,
    ):

        self.seed = seed
        self.loc = loc
        self.scale = scale
        self.n_samples = n_samples
        if truncation_set is None:
            truncation_set = [(-np.inf, np.inf)]
        self.lower_bound, self.upper_bound = self._get_truncation_bounds(truncation_set)
        self.interval_masses = np.array(
            [
                self._compute_mass_in_interval_avg(a, b)
                for a, b in zip(self.lower_bound, self.upper_bound)
            ]
        )
        super().__init__()

    def _pdf(self, x):  # pylint: disable=arguments-differ
        x = self._normalize(x)
        # n x 1 indicator that x is in an interval
        in_interval = np.array(
            [np.any((self.lower_bound <= x_i) & (x_i <= self.upper_bound)) for x_i in x]
        )
        return in_interval * norm.pdf(x) / (self.scale * self.interval_masses.sum())

    def _logpdf(self, x):  # pylint: disable=arguments-differ
        x = self._normalize(x)
        # n x 1 indicator that x is in an interval
        in_interval = np.array(
            [np.any((self.lower_bound <= x_i) & (x_i <= self.upper_bound)) for x_i in x]
        )
        logpdf = (
            norm.logpdf(x) - np.log(self.scale) - np.log(self.interval_masses.sum())
        )
        logpdf[~in_interval] = -np.inf
        return logpdf

    def _cdf(self, x):  # pylint: disable=arguments-differ
        x = self._normalize(x)

        if self.lower_bound.size == self.upper_bound.size == 0:
            # i.e., truncation set is empty
            return (x > 0).astype(float)

        denominator = self.interval_masses.sum()
        if denominator == 0:
            # converts cdf to 0 or 1 depending on bounds and x
            a = self.lower_bound.min()
            b = self.upper_bound.max()
            convert_0_1 = lambda x_i: a < x_i if x_i > 0 else b < x_i
            return np.array([convert_0_1(x_i) for x_i in x]).astype(float)

        return np.clip(self._compute_cdf_numerator(x) / denominator, 0, 1)

    def _logcdf(self, x):  # pylint: disable=arguments-differ
        x = self._normalize(x)
        denominator = self.interval_masses.sum()
        return np.log(self._compute_cdf_numerator(x)) - np.log(denominator)

    def _compute_mass_in_interval_avg(self, a, b):
        # compute the amount of mass in the interval [a, b] by averaging approximate
        # mass in [a, b] and [-b, -a]
        # the amount of mass in these intervals is the same because the normal is symmetric
        # this can improve performance
        arr = [
            self._compute_mass_in_interval(a, b, int(self.n_samples / 2)),
            self._compute_mass_in_interval(-b, -a, int(self.n_samples / 2)),
        ]
        return np.mean(arr, where=~np.isnan(arr))

    def _compute_mass_in_interval(self, a, b, n_samples=None):
        # compute the amount of mass in the interval [a, b] using minimax exponential tilt
        def compute_psi(params):
            x, mu = params
            return -x * mu + (0.5 * mu ** 2 + np.log(norm.cdf(b, mu) - norm.cdf(a, mu)))

        def d_psi_d_mu(params):
            # derivative of psi with respect to mu
            x, mu = params
            return (
                -x
                + mu
                + (norm.pdf(b, mu) - norm.pdf(a, mu))
                / ((norm.cdf(b, mu) - norm.cdf(a, mu)))
            )

        def optimize_params(x0):
            # maximize psi subject to x being contained in the interval and the
            # derivative of psi wrt mu =0 0
            derivative_constraint = NonlinearConstraint(d_psi_d_mu, 0, 0)
            return minimize(
                lambda x: -compute_psi(x),
                x0=x0,
                bounds=[(a, b), (-np.inf, np.inf)],
                constraints=[derivative_constraint],
            )

        def compute_tilting_param():
            # compute the optimal tilting parameter
            try:
                # initial guess for x, denoted as Psi in the paper
                # in the univariate case, the initial guess for mu is 0
                x_init = (norm.pdf(a) - norm.pdf(b)) / ((norm.cdf(b) - norm.cdf(a)))
            except ZeroDivisionError:
                x_init = a if a < 0 else b

            if a < x_init < b:
                # optimal x is in the truncation set, so optimal mu is 0
                return 0

            # optimal mu must be found by non-linear optimization
            res = optimize_params([x_init, 0])
            if res.success:
                return res.x[1]
            next_guess, final_guess = ([a, a], [b, b]) if a > 0 else ([b, b], [a, a])
            res = optimize_params(next_guess)
            if res.success:
                return res.x[1]
            res = optimize_params(final_guess)
            if res.success:
                return res.x[1]
            warnings.warn(
                "Optimizer failed to find truncated normal tilt parameter",
                RuntimeWarning,
            )
            return x_init

        if b < a:
            return 0
        mu = compute_tilting_param()
        x = truncnorm_base.rvs(
            a - mu, b - mu, mu, size=n_samples or self.n_samples, random_state=self.seed
        )
        return np.exp(compute_psi((x, mu))).mean()

    def _compute_cdf_numerator(self, x):
        # n x p indicates x is above the upper bound of the interval
        index = np.array([self.upper_bound <= x_i for x_i in x])
        # n x 1 CDF for the intervals where x is above the upper bound
        cdf = index @ self.interval_masses

        # tuple of (x index, interval index) such that x[x index] is in interval[interval_index]
        indices = np.where(
            [(self.lower_bound < x_i) & (x_i < self.upper_bound) for x_i in x]
        )
        # add mass from intervals containing x
        cdf[indices[0]] += np.array(
            [
                self._compute_mass_in_interval_avg(
                    self.lower_bound[interval_index], x[x_index]
                )
                for x_index, interval_index in zip(*indices)
            ]
        )

        return cdf

    def _get_truncation_bounds(self, truncation_set):
        if not truncation_set:
            return np.array([]), np.array([])

        for interval in truncation_set:
            if interval[1] < interval[0]:
                raise ValueError(f"Invalid interval {interval}")

        truncation_set.sort(key=lambda x: x[0])
        a, b = list(zip(*truncation_set))

        # ensure b is strictly increasing
        b = [b[0]] + [max(b_i, b_j) for b_i, b_j in zip(b[1:], b[:-1])]

        new_a, new_b = [a[0]], []
        for a_i, b_i in zip(a[1:], b[:-1]):
            if a_i > b_i:
                new_b.append(b_i)
                new_a.append(a_i)
        new_b.append(b[-1])

        return np.array(new_a), np.array(new_b)

    def _normalize(self, arr):
        return (arr - self.loc) / self.scale

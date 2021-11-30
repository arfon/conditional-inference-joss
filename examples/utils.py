"""Utilities for example notebooks.
"""
from __future__ import annotations

import copy
from itertools import chain

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import norm

from conditional_inference.stats import truncnorm


def confidence_ellipse(
    mean, cov, ax, stds=[1, 2, 3], palette=sns.color_palette(), **kwargs
):
    """
    Create a plot of the covariance confidence ellipse.

    Parameters:
        mean (np.ndarray): (2,) mean vector.
        cov (np.ndarray): (2, 2) covariance matrix.
        ax (matplotlib.axes.Axes): The axes object to draw the ellipse into.
        stds (list[float]): The number of standard deviations to determine the ellipse's radiuses.
        **kwargs (Any): Forwarded to `~matplotlib.patches.Ellipse`

    Returns:
        matplotlib.patches.Ellipse
    """
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius = np.sqrt(1 + pearson), np.sqrt(1 - pearson)

    ellipses = []
    for std in stds:
        # Calculating the standard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale = np.sqrt(cov[0, 0]) * std, np.sqrt(cov[1, 1]) * std
        transf = transforms.Affine2D().rotate_deg(45).scale(*scale).translate(*mean)
        ellipse = Ellipse(
            (0, 0),
            width=ell_radius[0] * 2,
            height=ell_radius[1] * 2,
            facecolor="none",
            edgecolor=palette[0],
            **kwargs,
        )
        ellipse.set_transform(transf + ax.transData)
        ellipses.append(ax.add_patch(ellipse))

    return ellipses


class RankConditionAnimation:
    """Rank condition animation helper class.

    Args:
        mean (np.ndarray): Vector of conventional estimates.
        cov (np.ndarray): Conventional covariance matrix.
        index (int): Index of parameter of interest.
        rank (list[int]): Conditional rank order.
        xlim (tuple[float, float]): Limits of the x-axis on the graph.
        palette (list, optional): Color palette. Defaults to sns.color_palette().
        n_frames (int, optional): Number of frames in the animation. Defaults to 120.
    """

    def __init__(
        self,
        mean: np.ndarray,
        cov: np.ndarray,
        index: int,
        rank: list[int],
        xlim: tuple[float, float],
        palette: list = sns.color_palette(),
        n_frames: int = 120,
    ):
        self._linspace = np.linspace(*xlim)
        self._init_func()

        self.mean = mean
        self.cov = cov
        # variance of conventional estimates
        # given the value of the conventional estimate of parameter[index]
        # based on conditional multivariate normal
        self.conditional_var = np.delete(
            np.diag(cov) - cov[index] ** 2 / cov[index, index], index
        )
        self.index = index
        self.rank = rank
        self.xlim = xlim
        # lower and upper values of the y-axis
        pdf_max = max([norm.pdf(0, 0, np.sqrt(var)) for var in self.conditional_var])
        self.ylim = (-0.1 * pdf_max, 1.1 * pdf_max)
        self.ymin = (0 - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        self.palette = palette
        self.n_frames = n_frames

    def _init_func(self) -> None:
        """Init function for animation."""
        if hasattr(self, "_truncation_sets"):
            [i.remove() for i in self._truncation_sets]
        self._truncation_sets = []
        self._prev_rank = None
        self._xmin = None

    def _animate(self, i: int) -> list:
        """Animation function.

        Args:
            i (int): Frame number.

        Returns:
            list: Arists.
        """

        def update_truncation_set(value):
            # extend the polygons highlighting the truncation set to ``value``
            self._truncation_sets[-1].set_xy(
                [
                    [self._xmin, self.ymin],
                    [self._xmin, 1],
                    [value, 1],
                    [value, self.ymin],
                    [self._xmin, self.ymin],
                ]
            )

        # update the conventional estimate of parameter[index]
        x = self.xlim[0] + i * (self.xlim[1] - self.xlim[0]) / self.n_frames
        # update the vertical line at the conventional estimate of parameter[index]
        self._conventional_vline.set_data([x, x], [self.ymin, 1])

        # compute the conventional point estimates
        # given the value of the conventional estimate of parameter[index]
        # and update distribution plots
        conditional_mean = np.delete(
            self.mean
            + self.cov[self.index]
            / (self.cov[self.index, self.index] ** 2)
            * (x - self.mean[self.index]),
            self.index,
        )
        for (dist_line, mean_line), mean, var in zip(
            self._distribution_plots, conditional_mean, self.conditional_var
        ):
            dist_line.set_data(
                self._linspace, norm.pdf(self._linspace, mean, np.sqrt(var))
            )
            mean_line.set_data([mean, mean], [self.ymin, 1])

        # update the current rank text
        current_rank = np.sum(x <= conditional_mean) + 1
        self._rank_text.set_text(f"Rank {current_rank}")

        if current_rank in self.rank:
            # update the vspace polygons highlighting the truncation set
            if self._prev_rank in self.rank:
                # extend the current polygon
                update_truncation_set(x)
            else:
                # start a new polygon
                try:
                    self._xmin = x
                except ValueError:
                    self._xmin = self.xlim[0]
                self._truncation_sets.append(
                    self._ax.axvspan(
                        self._xmin, self._xmin, color=self.palette[2], alpha=0.2
                    )
                )
        self._prev_rank = current_rank

        return (
            list(chain(self._distribution_plots))
            + self._truncation_sets
            + [self._conventional_vline, self._rank_text]
        )

    def make_animation(
        self, title: str = None, xlabel: str = None
    ) -> animation.FuncAnimation:
        """Make a rank condition animation.

        Args:
            title (str, optional): Title of the graph. Defaults to None.
            xlabel (str, optional): Label of the graph x-axis. Defaults to None.

        Returns:
            animation.FuncAnimation: Animation.
        """
        fig = plt.figure()
        self._ax = ax = fig.add_subplot(xlim=self.xlim, ylim=self.ylim)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # vertical line at the value of the conventional estimate of parameter[index]
        self._conventional_vline = ax.axvline(
            self.xlim[0], color=self.palette[1], linestyle="--"
        )
        # (distribution of conventional estimate line plot, vertical line at conventional point estimate) tuples
        self._distribution_plots = [
            (
                ax.plot([], [], color=self.palette[0])[0],
                ax.axvline(color=self.palette[0], linestyle="--"),
            )
            for _ in range(len(self.mean) - 1)
        ]
        # text displaying the rank of the conventional estimate of the effect of policy[index]
        self._rank_text = ax.text(
            self.xlim[0], self.ylim[0] + 0.02 * (self.ylim[1] - self.ylim[0]), ""
        )

        return animation.FuncAnimation(
            fig, self._animate, self.n_frames, init_func=self._init_func, blit=True
        )


class QuantileUnbiasedAnimation:
    """Quantile-unbiased animation helper class.

    Parameters:
        x (float): Conventional point estimate.
        scale (float): Standard deviation of the conventional estimate.
        truncation_set (list[tuple[float, float]]): Truncation set for the conditioning event.
        xlim (tuple[float, float]): Limits of the x-axis.
        projection_quantile (float): For use with projection CIs. Defaults to None.
        palette (list[color-like]): List of colors (passed to matplotlib functions).
            Defaults to seaborn default palette.
        n_frames (int): Number of frames to animate. Defaults to 120.
    """

    def __init__(
        self,
        x: float,
        scale: float,
        truncation_set: list[tuple[float, float]],
        xlim: tuple[float, float],
        projection_quantile: float = None,
        palette: list = sns.color_palette(),
        n_frames: int = 120,
    ):
        # y limits of the graph
        self._ylim = -0.1, 1.1
        # relative values of 0 and 1 on the y axis
        self._ymin, self._ymax = (np.array([0, 1]) - self._ylim[0]) / (
            self._ylim[1] - self._ylim[0]
        )
        # x data for the quantile-unbiased CDF plot
        self._x_data = []
        # y data for the quantile-unbiased CDF plot
        self._cdf_data = []
        # relative position of x on the x-axis
        self._x_relative = (x - xlim[0]) / (xlim[1] - xlim[0])
        self._linspace = np.linspace(xlim[0], xlim[1])

        self.truncation_set = truncation_set
        self.x = x
        self.scale = scale
        self.xlim = xlim
        self.projection_len = (
            None if projection_quantile is None else projection_quantile * scale
        )
        self.palette = palette
        self.n_frames = n_frames

    def _init_func(self) -> None:
        """Init function for animation."""
        self._x_data.clear()
        self._cdf_data.clear()

    def _animate(self, i: int) -> list:
        """Animation function.

        Args:
            i (int): Frame number.

        Returns:
            list: Artists.
        """
        # get the location parameter for the current frame
        loc = self.xlim[0] + i * (self.xlim[1] - self.xlim[0]) / self.n_frames

        # compute the trunction set
        if self.projection_len is None:
            truncation_set = copy.deepcopy(self.truncation_set)
        else:
            # take the intersection of the truncation set and the projection CI
            truncation_set = []
            for a, b in self.truncation_set:
                a, b = max(loc - self.projection_len, a), min(
                    loc + self.projection_len, b
                )
                if a < b:
                    truncation_set.append((a, b))
        # standardize the truncation set
        truncation_set = [
            ((lower - loc) / self.scale, (upper - loc) / self.scale)
            for lower, upper in truncation_set
        ]

        # update the vertical line at the location parameter
        self._loc_vline.set_data([loc, loc], [self._ymin, 1])

        # update the survival function plot
        truncnorm_dist = truncnorm(truncation_set, loc, self.scale)
        if truncation_set:
            self._sf_plot.set_data(self._linspace, truncnorm_dist.sf(self._linspace))
        else:
            # truncation set is empty, survival function is not well defined
            self._sf_plot.set_data([], [])

        # update the horizontal line at the quantile-unbiased CDF evaluated at loc
        cdf = truncnorm_dist.sf(self.x)
        # location parameter relative to x-limits of the graph
        loc_relative = (loc - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        self._cdf_hline.set_data(
            [min(self._x_relative, loc_relative), max(self._x_relative, loc_relative)],
            [cdf, cdf],
        )

        # update the quantile-unbiased CDF plot
        self._x_data.append(loc)
        self._cdf_data.append(cdf)
        self._cdf_plot.set_data(self._x_data, self._cdf_data)

        plots = [self._loc_vline, self._sf_plot, self._cdf_hline, self._cdf_plot]
        if self.projection_len is None:
            return plots

        # update the projection CI plot
        xmin = max(loc - self.projection_len, self.xlim[0])
        xmax = min(loc + self.projection_len, self.xlim[1])
        self._projection_plot.set_xy(
            [
                [xmin, self._ymin],
                [xmin, 1],
                [xmax, 1],
                [xmax, self._ymin],
                [xmin, self._ymin],
            ]
        )
        return plots + [self._projection_plot]

    def make_animation(
        self, title: str = None, xlabel: str = None
    ) -> animation.FuncAnimation:
        """Make a quantile-unbiased estimator animation.

        Args:
            title (str, optional): Graph title. Defaults to None.
            xlabel (str, optional): Graph x-axis label. Defaults to None.

        Returns:
            animation.FuncAnimation: Animation.
        """
        fig = plt.figure()
        ax = fig.add_subplot(xlim=self.xlim, ylim=self._ylim)
        ax.set_ylabel("alpha")
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)

        # draw a vertical line at the conventional estimate
        ax.axvline(self.x, self._ymin, color=self.palette[1], linestyle="--")
        ax.plot(
            self._linspace,
            norm.cdf(self._linspace, self.x, self.scale),
            color=self.palette[1],
        )

        # highlight the truncation set
        for xmin, xmax in self.truncation_set:
            ax.axvspan(
                max(xmin, self.xlim[0]),
                min(xmax, self.xlim[1]),
                ymin=self._ymin,
                color=self.palette[2],
                alpha=0.2,
            )

        # vertical line at the location parameter of the truncated normal
        self._loc_vline = ax.axvline(
            self.xlim[0], color=self.palette[3], linestyle="--"
        )
        # plot of the survival function of the truncated normal
        (self._sf_plot,) = ax.plot([], [], color=self.palette[3])
        # horizontal line at the survival function evaluated at the conventional estimate
        self._cdf_hline = ax.axhline(color=self.palette[4], linestyle="--")
        # plot of the quantile-unbiased CDF
        (self._cdf_plot,) = ax.plot([], [], color=self.palette[4])
        # highlight the projection confidence set
        self._projection_plot = None
        if self.projection_len is not None:
            self._projection_plot = ax.axvspan(
                0, 0, color=self.palette[5], ymin=self._ymin, alpha=0.2
            )

        return animation.FuncAnimation(
            fig, self._animate, self.n_frames, init_func=self._init_func, blit=True
        )

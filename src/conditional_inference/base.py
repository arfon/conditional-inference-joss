"""Base classes.
"""
from __future__ import annotations

import pickle
from typing import Any, List, Sequence, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable

# TODO: remove type ignore statements for starred arguments when this issue is fixed
# https://github.com/python/mypy/issues/6799

ColumnType = Union[str, int]
ColumnsType = Sequence[ColumnType]
ModelType = TypeVar("ModelType", bound="ModelBase")
Numeric1DArray = Sequence[float]
ResultsType = TypeVar("ResultsType", bound="ResultsBase")


class ConventionalEstimatesData:
    """Data store for conventional estimates.

    Args:
        mean (Numeric1DArray): (n,) array of means.
        cov (np.ndarray): (n,n) covariance matrix.
        endog_names (str, optional): Name of endogenous variable. Defaults to None.
        exog_names (Sequence[str], optional): Name of exogenous variables. Defaults to None.

    Attributes:
        mean (Numeric1DArray): (n,) array of means.
        cov (np.ndarray): (n,n) covariance matrix.
        endog_names (str, optional): Name of endogenous variable.
        exog_names (Sequence[str], optional): Name of exogenous variables.
    """

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        endog_names: str = None,
        exog_names: Sequence[str] = None,
    ):
        self.mean_orig = self.mean = mean
        self.cov = cov * np.identity(len(mean)) if np.isscalar(cov) else cov
        self.endog_names = endog_names
        self.exog_names = exog_names

    @property
    def mean(self):  # pylint: disable=missing-function-docstring
        return self._mean

    @mean.setter
    def mean(self, mean: Numeric1DArray):  # pylint: disable=missing-function-docstring
        self._mean = np.atleast_1d(mean)

    @property
    def endog_names(self):  # pylint: disable=missing-function-docstring
        return "y" if self._endog_names is None else self._endog_names

    @endog_names.setter
    def endog_names(
        self, endog_names: str
    ):  # pylint: disable=missing-function-docstring
        self._endog_names = endog_names

    @property
    def exog_names(self):  # pylint: disable=missing-function-docstring
        if self._exog_names is not None:
            return self._exog_names
        if hasattr(self.mean_orig, "index") and hasattr(
            self.mean_orig.index, "to_list"
        ):
            # assume mean is pd.Series-like
            return self.mean_orig.index.to_list()
        zfill = int(np.log10(len(self.mean)))
        return [f"x{str(i).zfill(zfill)}" for i in range(len(self.mean))]

    @exog_names.setter
    def exog_names(
        self, exog_names: Sequence[str]
    ):  # pylint: disable=missing-function-docstring
        self._exog_names = exog_names


class ModelBase:
    """Base for model classes.

    Args:
        mean (Numeric1DArray): (n,) array of means from conventional estimation.
        cov (np.ndarray): (n, n) covariance matrix.
        *args (Any): Passed to :class:`ConventionalEstimatesData`.
        seed (int, optional): Random seed. Defaults to 0.
        **kwargs (Any): Passed to :class:`ConventionalEstimatesData`.

    Attributes:
        data (ConventionalEstimatesData): Conventional estimates data.
        seed (int): Random seed.

    Notes:
        Properties of :class:`ConventionalEstimatesData` can be accessed directly, e.g.,

        .. doctest::

            >>> from conditional_inference.base import ModelBase
            >>> import numpy as np
            >>> model = ModelBase([1, 2, 3], np.identity(3))
            >>> model.mean
            array([1, 2, 3])
    """

    _data_properties = [
        "mean",
        "cov",
        "endog_names",
        "exog_names",
    ]

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        *args: Any,
        seed: int = 0,
        **kwargs: Any,
    ):
        self.data = ConventionalEstimatesData(mean, cov, *args, **kwargs)
        self.seed = seed

    def __getattribute__(self, key):
        if key != "_data_properties" and key in self._data_properties:
            return getattr(self.data, key)
        return super().__getattribute__(key)

    def __setattr__(self, key, val):
        if key in self._data_properties:
            setattr(self.data, key, val)
        else:
            super().__setattr__(key, val)

    @classmethod
    def from_results(
        cls: Type[ModelType],
        results: LikelihoodModelResults,
        *args,
        cols: ColumnsType = None,
        **kwargs,
    ) -> ModelType:
        """Instantiate an estimator from conventional regression results.

        Args:
            results (LikelihoodModelResults): Conventional likelihood model estimates.
            *args (Any): Passed to the model class constructor.
            cols (ColumnsType, optional): Names or indices of the policy variables. Defaults to
                None.
            **kwargs (Any): Passed to the model class constructor.

        Returns:
            Model: Estimator.

        Examples:

            .. code-block::

                >>> from conditional_inference.base import ModelBase
                >>> import numpy as np
                >>> import statsmodels.api as sm
                >>> X = np.repeat(np.identity(3), 100, axis=0)
                >>> beta = np.array([0, 1, 2])
                >>> y = X @ beta + np.random.normal(size=300)
                >>> ols_results = sm.OLS(y, X).fit()
                >>> model = ModelBase.from_results(ols_results)
                >>> model.mean
                array([-0.20434022,  0.96700821,  1.88196662])
                >>> model.cov
                array([[0.01163716, 0.        , 0.        ],
                       [0.        , 0.01163716, 0.        ],
                       [0.        , 0.        , 0.01163716]])
        """

        def get_index(col: Union[str, int]) -> int:
            if isinstance(col, str):
                return results.model.exog_names.index(col)
            if np.isscalar(col):
                return int(col)
            raise ValueError(
                f"Invalid column type {type(col)} for column {col}"
            )  # pragma: no cover

        if cols is None:
            indices = np.arange(results.params.shape[0])
            exog_names = results.model.exog_names
        else:
            indices = np.array([get_index(col) for col in cols])
            exog_names = [results.model.exog_names[i] for i in indices]

        cov = results.cov_params()
        if isinstance(cov, pd.DataFrame):
            cov = cov.values

        return cls(
            pd.Series(results.params[indices], index=exog_names),
            cov[indices][:, indices],
            endog_names=kwargs.pop("endog_names", results.model.endog_names),
            *args,
            **kwargs,
        )

    @classmethod
    def from_csv(
        cls: Type[ModelType],
        filename: str,
        *args: Any,
        cols: ColumnsType = None,
        **kwargs: Any,
    ) -> ModelType:
        """Instantiate an estimator from csv file.

        Args:
            filename (str): Name of the csv file.
            *args (Any): Passed to the model class constructor.
            cols (ColumnsType, optional): Names or indices of the policy variables. Defaults to
                None.
            **kwargs (Any): Passed to the model class constructor.

        Returns:
            Model: Estimator.
        """

        def get_index(col: Union[str, int]) -> int:
            if isinstance(col, str):
                return exog_names.index(col)
            if np.isscalar(col):
                return int(col)
            raise ValueError(
                f"Invalid column type {type(col)} for column {col}"
            )  # pragma: no cover

        df = pd.read_csv(filename)
        mean, cov = df.values[:, 0], df.values[:, 1:]  # pylint: disable=no-member
        endog_names, exog_names = (
            df.columns[0],  # pylint: disable=no-member
            df.columns[1:],  # pylint: disable=no-member
        )

        # select columns
        if cols is None:
            indices = np.arange(len(df))
        else:
            indices = np.array([get_index(col) for col in cols])
            exog_names = [exog_names[i] for i in indices]

        return cls(
            pd.Series(mean[indices], index=exog_names),
            cov[indices][:, indices],
            endog_names=kwargs.pop("endog_names", endog_names),
            *args,
            **kwargs,
        )

    def get_indices(self, cols: ColumnsType = None) -> np.ndarray:
        """Get indices associated with columns.

        Args:
            cols (ColumnsType, optional): Column names or indices. Defaults to None.

        Returns:
            np.ndarray: Indices of requested columns.
        """
        if cols is None:
            return np.arange(self.mean.shape[0])

        if isinstance(cols, str):
            if cols == "sorted":
                return (-self.mean).argsort()
            return np.array([self._get_index(cols)])

        return np.array([self._get_index(col) for col in cols])

    def _get_index(self, col: ColumnType) -> int:
        return self.exog_names.index(col) if isinstance(col, str) else col


class ResultsBase:
    """Base for results classes.

    Args:
        model (ModelBase): Model on which the results are based.
        cols (ColumnsType, optional): Columns of interest. Defaults to None.
        title (str, optional): Results title. Defaults to "Estimation results".
    """

    def __init__(
        self,
        model: ModelBase,
        cols: ColumnsType = None,
        title: str = "Estimation results",
    ):
        self.model = model
        self.indices = model.get_indices(cols)
        self.n_policies = len(self.indices)
        self.title = title

    def conf_int(self, alpha: float = 0.05, cols: ColumnsType = None) -> np.ndarray:
        """Compute the 1-alpha confidence interval.

        Args:
            alpha (float, optional): The CI will cover the truth with probability 1-alpha. Defaults
                to 0.05.
            cols (ColumnsType, optional): Names or indices of policies of interest. Defaults to
                None.

        Returns:
            np.ndarray: (n,2) array of confidence intervals.
        """
        if not hasattr(self, "distributions"):
            raise AttributeError(
                "Results object does not have `distributions` attribute."
            )

        if not hasattr(self, "params"):
            raise AttributeError("Results object does not have `params` attribute.")

        indices = self._get_indices(cols)
        return np.array(
            [
                dist.ppf([alpha / 2, 1 - alpha / 2])
                for index, dist in enumerate(self.distributions)  # type: ignore, pylint: disable=no-member
                if index in indices
            ]
        )

    def point_plot(
        self,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
        alpha: float = 0.05,
        ax=None,
    ):
        """Create a point plot.

        Args:
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): Names of the policies. Defaults to None.
            title (str, optional): Plot title. Defaults to None.
            alpha: (float, optional): Plot the 1-alpha CI. Defaults to 0.05.
            ax: (AxesSubplot, optional): Axis to write on.

        Returns:
            plt.axes._subplots.AxesSubplot: Plot.
        """
        if not hasattr(self, "params"):
            raise AttributeError("Results object does not have `params` attribute.")

        conf_int = self.conf_int(alpha)
        xname = xname or [self.model.exog_names[idx] for idx in self.indices]
        yticks = np.arange(len(xname), 0, -1)

        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(
            x=self.params,  # type: ignore, pylint: disable=no-member
            y=yticks,
            xerr=[self.params - conf_int[:, 0], conf_int[:, 1] - self.params],  # type: ignore, pylint: disable=no-member
            fmt="o",
        )
        ax.set_title(title or self.title)
        ax.set_xlabel(yname or self.model.endog_names)
        ax.set_yticks(yticks)
        ax.set_yticklabels(xname)

        return ax

    def save(self: ResultsType, fname: str) -> ResultsType:
        """Pickle results.

        Args:
            fname (str): File name.

        Returns:
            ResultsType: self.
        """
        with open(fname, "wb") as results_file:
            pickle.dump(self, results_file)
        return self

    def summary(
        self,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
        alpha: float = 0.05
    ) -> Summary:
        """Create a summary table.

        Args:
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): Names of the exogenous variables. Defaults
                to None.
            title (str, optional): Table title. Defaults to None.
            alpha (float, optional): Display 1-alpha confidence interval. Defaults to
                0.05.

        Returns:
            Summary: Summary table.
        """
        if not hasattr(self, "params"):
            raise AttributeError("Results object does not have `params` attribute.")

        if not hasattr(self, "pvalues"):
            raise AttributeError("Results object does not have `pvalues` attribute.")

        params_header = self._make_summary_header(alpha)
        params_data = np.hstack(
            (np.array([self.params, self.pvalues]).T, self.conf_int(alpha))  # type: ignore, pylint: disable=no-member
        )
        return self._make_summary(
            params_header,
            params_data,
            yname=yname,
            xname=xname,
            title=title,
        )

    def _get_indices(self, cols: ColumnsType = None) -> np.array:
        if not hasattr(self, "params"):
            raise AttributeError(
                f"Results object {self.__class__.__qualname__} has no attribute `params`"
            )
        return (
            np.arange(len(self.params))  # type: ignore, pylint: disable=no-member
            if cols is None
            else self.model.get_indices(cols)
        )

    def _make_summary(
        self,
        params_header: List[str],
        params_data: np.ndarray,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
    ) -> Summary:
        """Create a summary table.

        Args:
            params_header (List[str]): Table header
            params_data (np.ndarray): Table data.
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): Names of the exogenous variables. Defaults to None.
            title (str, optional): Table title. Defaults to None.

        Returns:
            Summary: Summary table.
        """
        params_stubs = xname or [self.model.exog_names[idx] for idx in self.indices]
        params_data_str = [[f"{val:.3f}" for val in row] for row in params_data]

        smry = Summary()
        smry.tables = [
            SimpleTable(
                params_data_str,
                params_header,
                params_stubs,
                title=title or self.title,
            ),
            SimpleTable(
                [[yname or self.model.endog_names]],
                stubs=["Dep. Variable"],
            ),
        ]

        return smry

    def _make_summary_header(self, alpha: float) -> List[str]:
        # make the header for the summary table
        # when subclassing ResultsBase, you may wish to overwrite this method
        return ["coef", "pvalue", f"[{alpha/2}", f"{1-alpha/2}]"]

"""Base classes.
"""
from __future__ import annotations

import pickle
from typing import Any, List, Optional, Sequence, Type, TypeVar, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.base.model import LikelihoodModelResults
from statsmodels.iolib.summary import Summary
from statsmodels.iolib.table import SimpleTable

# TODO: remove type ignore statements for starred arguments when this issue is fixed
# https://github.com/python/mypy/issues/6799

ColumnType = Union[str, int]
ColumnsType = Union[Sequence[int], Sequence[str], Sequence[bool]]
ModelType = TypeVar("ModelType", bound="ModelBase")
Numeric1DArray = Sequence[float]
ResultsType = TypeVar("ResultsType", bound="ResultsBase")


class ResultsBase:
    """Base for results classes.

    Args:
        model (ModelBase): Model on which the results are based.
        title (str, optional): Results title. Defaults to "Estimation results".
    """

    _default_title = "Estimation results"

    def __init__(
        self,
        model: ModelType,
        title: str = None,
    ):
        self.model = model
        if not hasattr(self, "pvalues"):
            self.pvalues = np.full(len(model.mean), np.nan)
        self.title = title
        self._conf_int_cached = {}

    @property
    def title(self) -> str:
        return self._title or self._default_title

    @title.setter
    def title(self, title: str) -> None:
        self._title = title

    def conf_int(
        self, alpha: float = 0.05, columns: ColumnsType = None, **kwargs: Any
    ) -> np.ndarray:
        """Compute the 1-alpha confidence interval.

        Args:
            alpha (float, optional): The CI will cover the truth with probability
                1-alpha. Defaults to 0.05.
            columns (ColumnsType, optional): Selected columns. Defaults to None.

        Returns:
            np.ndarray: (# params, 2) array of confidence intervals.
        """
        return self._conf_int(alpha, self.model.get_indices(columns), **kwargs)

    def _conf_int(self, alpha: float, indices: np.array) -> np.ndarray:
        if not hasattr(self, "marginal_distributions"):
            raise AttributeError(
                "Results object does not have `marginal_distributions` attribute."
            )

        return np.array(
            [
                self.marginal_distributions[i].ppf([alpha / 2, 1 - alpha / 2])
                for i in indices
            ]
        )

    def point_plot(
        self,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
        alpha: float = 0.05,
        columns: ColumnsType = None,
        spacing: float = 1,
        ax=None,
    ):
        """Create a point plot.

        Args:
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): (# params,) sequence of parameter names.
                Defaults to None.
            title (str, optional): Plot title. Defaults to None.
            alpha (float, optional): Plot the 1-alpha CI. Defaults to 0.05.
            columns (ColumnsType, optional): Selected columns. Defaults to None.
            spacing (float): Spacing on the horizontal axis. Defaults to 1.
            ax: (AxesSubplot, optional): Axis to write on.

        Returns:
            AxesSubplot: Plot.
        """

        if not hasattr(self, "params"):
            raise AttributeError("Results object does not have `params` attribute.")

        indices = self.model.get_indices(columns)
        params = self.params[indices]
        conf_int = self.conf_int(alpha, columns)
        yticks = spacing * np.arange(len(indices), 0, -1)

        if ax is None:
            _, ax = plt.subplots()
        ax.errorbar(
            x=params,  # type: ignore, pylint: disable=no-member
            y=yticks,
            xerr=[params - conf_int[:, 0], conf_int[:, 1] - params],  # type: ignore, pylint: disable=no-member
            fmt="o",
        )
        ax.set_title(title or self.title)
        ax.set_xlabel(yname or self.model.endog_names)
        ax.set_yticks(yticks)
        ax.set_yticklabels(self.model.exog_names[indices] if xname is None else xname)

        return ax

    def save(self: ResultsType, filename: str) -> None:
        """Pickle results.

        Args:
            filename (str): File name.
        """
        with open(filename, "wb") as results_file:
            pickle.dump(self, results_file)

    def summary(
        self,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
        alpha: float = 0.05,
        columns: ColumnsType = None,
    ) -> Summary:
        """Create a summary table.

        Args:
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): Names of the exogenous variables. Defaults
                to None.
            title (str, optional): Table title. Defaults to None.
            alpha (float, optional): Display 1-alpha confidence interval. Defaults to
                0.05.
            columns (ColumnsType, optional): Selected columns. Defaults to None.

        Returns:
            Summary: Summary table.
        """
        if not hasattr(self, "params"):
            raise AttributeError("Results object does not have `params` attribute.")

        indices = self.model.get_indices(columns)
        params_header = self._make_summary_header(alpha)
        params_data = np.hstack(
            (np.array([self.params, self.pvalues]).T[indices], self.conf_int(alpha, columns))  # type: ignore, pylint: disable=no-member
        )
        return self._make_summary(
            params_header,
            params_data,
            yname=yname,
            xname=self.model.exog_names[indices] if xname is None else xname,
            title=title,
        )

    def _make_summary(
        self,
        params_header: list[str],
        params_data: np.ndarray,
        yname: str = None,
        xname: Sequence[str] = None,
        title: str = None,
    ) -> Summary:
        """Create a summary table.

        Args:
            params_header (list[str]): Table header
            params_data (np.ndarray): Table data.
            yname (str, optional): Name of the endogenous variable. Defaults to None.
            xname (Sequence[str], optional): Names of the exogenous variables. Defaults to None.
            title (str, optional): Table title. Defaults to None.

        Returns:
            Summary: Summary table.
        """
        params_stubs = list(self.model.exog_names if xname is None else xname)
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

    def _make_summary_header(self, alpha: float) -> list[str]:
        # make the header for the summary table
        # when subclassing ResultsBase, you may wish to overwrite this method
        return ["coef", "pvalue", f"[{alpha/2}", f"{1-alpha/2}]"]


class ModelBase:
    """Base for model classes.

    Args:
        mean (Numeric1DArray): (# params,) array of conventionally estimated means.
        cov (np.ndarray): (# params, # params) covariance matrix.
        X (np.ndarray, optional): (# params, # features) feature matrix. Defaults to
            None.
        endog_names (str, optional): Name of endogenous variable. Defaults to None.
        exog_names (Sequence[str], optional): Names of the exogenous variables. Defaults
            to None.
        columns (ColumnsType, optional): Columns to use. This can be a sequence of
            indices (int), parameter names (str), or a Boolean mask. Defaults to None.
        sort (bool, optional): Sort the parameters by the conventionally estimated
            mean. Defaults to False.
        seed (int, optional): Random seed. Defaults to 0.

    Attributes:
        n_params (int): Number of estimated parameters.
        mean (np.ndarray): (# params,) array of conventionally estimated means.
        cov (np.ndarray): (# params, # params) covariance matrix.
        X (np.ndarray): (# params, # features) feature matrix.
        endog_names (str): Name of the endogenous variable.
        exog_names (np.ndarray): Name of exogenous variables.
        seed (int): Random seed.
    """

    _results_cls = ResultsBase

    def __init__(
        self,
        mean: Numeric1DArray,
        cov: np.ndarray,
        X: np.ndarray = None,
        endog_names: str = None,
        exog_names: Sequence[str] = None,
        columns: ColumnsType = None,
        sort: bool = False,
        random_state: int = 0,
    ):
        self.mean = mean
        self.n_params = len(self.mean) if columns is None else len(columns)
        self.cov = cov * np.identity(self.n_params) if np.isscalar(cov) else cov
        self.X = np.ones((len(self.mean), 1)) if X is None else X
        self.endog_names = endog_names
        if exog_names is None and isinstance(mean, pd.Series):
            self.exog_names = np.array(mean.index)
        else:
            self.exog_names = exog_names
        self.random_state = random_state

        # select columns
        indices = self.get_indices(columns)
        self.mean = self.mean[indices]
        self.cov = self.cov[indices][:, indices]
        self.X = self.X[indices]
        if exog_names is not None or isinstance(mean, pd.Series):
            self.exog_names = self.exog_names[indices]

        # sort columns
        if sort:
            argsort = (-self.mean).argsort()
            self.mean = self.mean[argsort]
            self.cov = self.cov[argsort][:, argsort]
            self.X = self.X[argsort]
            if exog_names is not None or isinstance(mean, pd.Series):
                self.exog_names = self.exog_names[argsort]

    @property
    def mean(self) -> np.ndarray:  # pylint: disable=missing-function-docstring
        return self._mean

    @mean.setter
    def mean(
        self, mean: Numeric1DArray
    ) -> None:  # pylint: disable=missing-function-docstring
        self._mean = np.atleast_1d(mean)

    @property
    def cov(self) -> np.ndarray:
        return self._cov

    @cov.setter
    def cov(self, cov: np.ndarray) -> None:
        self._cov = np.atleast_2d(cov)

    @property
    def endog_names(self) -> str:  # pylint: disable=missing-function-docstring
        return "y" if self._endog_names is None else self._endog_names

    @endog_names.setter
    def endog_names(
        self, endog_names: str
    ) -> None:  # pylint: disable=missing-function-docstring
        self._endog_names = endog_names

    @property
    def exog_names(self) -> np.ndarray:  # pylint: disable=missing-function-docstring
        if self._exog_names is not None:
            return self._exog_names
        zfill = int(np.log10(max(1, len(self.mean) - 1))) + 1
        return np.array([f"x{str(i).zfill(zfill)}" for i in range(len(self.mean))])

    @exog_names.setter
    def exog_names(
        self, exog_names: Optional[Sequence[str]]
    ) -> None:  # pylint: disable=missing-function-docstring
        self._exog_names = None if exog_names is None else np.atleast_1d(exog_names)

    @classmethod
    def from_results(
        cls: Type[ModelType],
        results: LikelihoodModelResults,
        **kwargs: Any,
    ) -> ModelType:
        """Initialize an estimator from conventional regression results.

        Args:
            results (LikelihoodModelResults): Conventional estimation results.
            **kwargs (Any): Passed to the model class constructor.

        Returns:
            Model: Estimator.

        Examples:

            .. testcode::

                import numpy as np
                import pandas as pd
                import statsmodels.api as sm
                from conditional_inference.base import ModelBase

                X = np.repeat(np.identity(3), 100, axis=0)
                beta = np.arange(3)
                y = X @ beta + np.random.normal(size=300)
                ols_results = sm.OLS(y, X).fit()
                model = ModelBase.from_results(ols_results)
                print(model.mean)
                print(model.cov)

            .. testoutput::

                [0.05980802 1.08201297 1.94076774]
                [[0.01007633 0.         0.        ]
                 [0.         0.01007633 0.        ]
                 [0.         0.         0.01007633]]
        """
        cov = results.cov_params()
        if isinstance(cov, pd.DataFrame):
            cov = cov.values

        return cls(
            results.params,
            cov,
            endog_names=kwargs.pop("endog_names", results.model.endog_names),
            exog_names=kwargs.pop("exog_names", results.model.exog_names),
            **kwargs,
        )

    @classmethod
    def from_csv(
        cls: Type[ModelType],
        filename: str,
        **kwargs: Any,
    ) -> ModelType:
        """Instantiate an estimator from csv file.

        Args:
            filename (str): Name of the csv file.
            **kwargs (Any): Passed to the model class constructor.

        Returns:
            Model: Estimator.
        """
        df = pd.read_csv(filename)
        mean, cov = df.values[:, 0], df.values[:, 1:]  # pylint: disable=no-member
        endog_names, exog_names = (
            df.columns[0],  # pylint: disable=no-member
            df.columns[1:],  # pylint: disable=no-member
        )

        return cls(
            mean,
            cov,
            endog_names=kwargs.pop("endog_names", endog_names),
            exog_names=kwargs.pop("exog_names", exog_names),
            **kwargs,
        )

    def to_csv(self, filename: str) -> None:
        """Write data to a csv.

        Args:
            filename (str): Name of the file to write to.
        """
        pd.DataFrame(
            np.hstack((self.mean.reshape(-1, 1), self.cov)),
            columns=[self.endog_names] + list(self.exog_names),
        ).to_csv(filename, index=False)

    def fit(self, *args: Any, **kwargs: Any) -> ResultsType:
        """Fit the model.

        Args:
            *args (Any): Passed to the results class constructor.
            **kwargs (Any): Passed to the results class constructor.

        Returns:
            ResultsType: Results.
        """
        return self._results_cls(self, *args, **kwargs)

    def get_index(self, column: ColumnType, names: Sequence[str] = None) -> int:
        """Get the index of a selected column.

        Args:
            column (ColumnType): Index or name of selected column.
            names (Sequence[str], optional): (# params,) sequence of names to select
                from.

        Returns:
            int: Index.
        """
        try:
            return int(column)
        except:
            pass

        return list(self.exog_names if names is None else names).index(column)

    def get_indices(
        self, columns: ColumnsType = None, names: Sequence[str] = None
    ) -> np.ndarray:
        """Get indices of the selected columns.

        Args:
            columns (ColumnsType, optional): Sequence of columns to select. The
            sequence can be a (# selected params,) sequence of column names (str) or
            indices (int), or a (# params,) boolean mask. Defaults to None.
            names (Sequence[str], optional): (# params,) sequence of names to select
                from.

        Returns:
            np.ndarray: (# selected params,) array of indices.
        """
        if names is None:
            names = self.exog_names

        if columns is None:
            return np.arange(len(names)).astype(int)

        cols = np.atleast_1d(columns)
        if cols.dtype == np.dtype("float"):
            cols = cols.astype(int)

        if cols.dtype == np.dtype(int):
            # cols is a sequence of indices
            return cols

        if cols.dtype == np.dtype(bool):
            # cols is a boolean mask
            return np.where(cols)[0]

        # cols are the parameter names (exogenous variables)
        sorter = np.argsort(names)
        return sorter[np.searchsorted(names, cols, sorter=sorter)]

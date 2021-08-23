import os
import pickle

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import norm

from conditional_inference.base import ModelBase, ResultsBase

tol = 0.001

n_policies = 3
mean = np.arange(n_policies)
cov = np.identity(n_policies)
model = ModelBase(mean, cov)
results = ResultsBase(model)


class TestData:
    @pytest.mark.parametrize("endog_names", [None, "target"])
    def test_endog_names(self, endog_names):
        # test that the model has the correct endogenous variable name
        model = ModelBase(mean, cov, endog_names=endog_names)
        if endog_names is None:
            assert model.endog_names == "y"
        else:
            assert model.endog_names == endog_names

    @pytest.mark.parametrize(
        "exog_names,index",
        [
            ([f"var{i}" for i in range(n_policies)], False),
            ([f"var{i}" for i in range(n_policies)], True),
            (None, False),
        ],
    )
    def test_exog_names(self, exog_names, index):
        # test that the model has the correct exogenous variable names
        if exog_names is None:
            model = ModelBase(mean, cov)
            assert model.exog_names == [f"x{i}" for i in range(n_policies)]
        else:
            if index:
                model = ModelBase(pd.Series(mean, index=exog_names), cov)
            else:
                model = ModelBase(mean, cov, exog_names=exog_names)
            assert model.exog_names == exog_names

    def test_set_attr(self):
        # test that you can set a data attribute by setting the model's same-named attribute
        model = ModelBase(mean, cov)
        exog_names = [f"var{i}" for i in range(n_policies)]
        model.exog_names = exog_names
        assert model.exog_names == exog_names
        assert model.data.exog_names == exog_names


@pytest.fixture(scope="module", params=[True, False])
def ols_results(
    request,
    n_obs_per_policy=100,
    exog_names=[f"var{i}" for i in range(n_policies)],
    endog_name="target",
):
    # create statsmodels OLS results
    X = pd.DataFrame(
        np.repeat(np.identity(n_policies), n_obs_per_policy, axis=0), columns=exog_names
    )
    y = X @ np.arange(n_policies) + norm.rvs(size=n_policies * n_obs_per_policy)
    y = pd.Series(y, name=endog_name)
    ols_results = sm.OLS(y, X).fit()
    return ols_results if request.param else ols_results.get_robustcov_results()


class TestModel:
    def get_params_cov(self, ols_results):
        # return the OLS point estimates and covariance matrix from results object
        params = ols_results.params
        params = params.values if isinstance(params, pd.Series) else params
        cov = ols_results.cov_params()
        cov = cov.values if isinstance(cov, pd.DataFrame) else cov
        return params, cov

    def compare_model_to_ols_results(self, model, ols_results):
        # make sure the model's attributes match those of the OLS results
        params, cov = self.get_params_cov(ols_results)
        assert ((model.mean - params) ** 2).mean() <= tol
        assert ((model.cov - cov) ** 2).mean() <= tol
        assert model.exog_names == ols_results.model.exog_names
        assert model.endog_names == ols_results.model.endog_names

    @pytest.mark.parametrize("cols", [None, "from_results", [0, 1, 2]])
    def test_from_results(self, ols_results, cols):
        # test that you can initialize a model from statsmodels results object
        if cols == "from_results":
            cols = ols_results.model.exog_names
        model = ModelBase.from_results(ols_results, cols=cols)
        self.compare_model_to_ols_results(model, ols_results)

    def test_from_csv(self, ols_results, filename="temp.csv"):
        # test that you can initialize a model from a csv file
        params, cov = self.get_params_cov(ols_results)
        df = pd.DataFrame(
            np.hstack((params.reshape(-1, 1), cov)),
            columns=[ols_results.model.endog_names] + ols_results.model.exog_names,
        )
        df.to_csv(filename, index=False)
        model = ModelBase.from_csv(filename)
        os.remove(filename)
        self.compare_model_to_ols_results(model, ols_results)

    @pytest.mark.parametrize(
        "cols", [None, "sorted", "x0", [f"x{i}" for i in range(n_policies-1, -1, -1)], [2, 1, 0]]
    )
    def test_get_indices(self, cols):
        indices = ModelBase(mean, cov).get_indices(cols)
        if cols is None:
            assert (indices == [0, 1, 2]).all()
        elif cols == "sorted":
            assert (indices == (-mean).argsort()).all()
        elif cols == "x0":
            assert (indices == [0]).all()
        else:  # columns are in reverse order
            assert (indices == [2, 1, 0]).all()


class TestResults:
    def test_conf_int(self):
        with pytest.raises(AttributeError):
            results.conf_int()

    def test_save(self, filename="temp.p"):
        results.save(filename)
        with open(filename, "rb") as results_file:
            loaded_results = pickle.load(results_file)
        os.remove(filename)
        assert (loaded_results.model.mean == results.model.mean).all()

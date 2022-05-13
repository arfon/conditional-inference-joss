import io
import os
import pickle

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.stats import norm

from conditional_inference.base import ModelBase

N_POLICIES = 3


@pytest.fixture(scope="module", params=[True, False])
def ols_results(
    request,
    n_obs_per_policy=100,
    exog_names=[f"var{i}" for i in range(N_POLICIES)],
    endog_name="target",
):
    # create statsmodels OLS results
    X = pd.DataFrame(
        np.repeat(np.identity(N_POLICIES), n_obs_per_policy, axis=0), columns=exog_names
    )
    y = X @ np.arange(N_POLICIES) + norm.rvs(size=N_POLICIES * n_obs_per_policy)
    y = pd.Series(y, name=endog_name)
    ols_results = sm.OLS(y, X).fit()
    return ols_results if request.param else ols_results.get_robustcov_results()


class TestModelBase:
    @pytest.mark.parametrize(
        "mean", ([0, 1, 2], pd.Series([0, 1, 2], index=["world", "moon", "star"]))
    )
    @pytest.mark.parametrize("exog_names", (None, ["mars", "europa", "sun"]))
    @pytest.mark.parametrize("sort", (True, False))
    def test__init__(self, mean, exog_names, sort):
        model = ModelBase(mean, np.diag([1, 2, 3]), exog_names=exog_names, sort=sort)

        expected_exog_names = ["x0", "x1", "x2"]

        if sort:
            expected_mean = [2, 1, 0]
            expected_cov = np.diag([3, 2, 1])
            if isinstance(mean, pd.Series):
                expected_exog_names = ["star", "moon", "world"]
            if exog_names is not None:
                expected_exog_names = ["sun", "europa", "mars"]
        else:
            expected_mean = [0, 1, 2]
            expected_cov = np.diag([1, 2, 3])
            if isinstance(mean, pd.Series):
                expected_exog_names = ["world", "moon", "star"]
            if exog_names is not None:
                expected_exog_names = ["mars", "europa", "sun"]

        np.testing.assert_array_equal(model.mean, expected_mean)
        np.testing.assert_array_equal(model.cov, expected_cov)
        np.testing.assert_array_equal(model.exog_names, expected_exog_names)

    @pytest.mark.parametrize(
        "columns", ([0.0, 2.0], [0, 2], ["world", "star"], [True, False, True])
    )
    @pytest.mark.parametrize("sort", (True, False))
    def test_column_selection(self, columns, sort):
        model = ModelBase(
            pd.Series([0, 1, 2], index=["world", "moon", "star"]),
            np.diag([1, 2, 3]),
            columns=columns,
            sort=sort,
        )

        if sort:
            expected_mean = [2, 0]
            expected_cov = np.diag([3, 1])
            expected_exog_names = ["star", "world"]
        else:
            expected_mean = [0, 2]
            expected_cov = np.diag([1, 3])
            expected_exog_names = ["world", "star"]
        np.testing.assert_array_equal(model.mean, expected_mean)
        np.testing.assert_array_equal(model.cov, expected_cov)
        np.testing.assert_array_equal(model.exog_names, expected_exog_names)

    @pytest.mark.parametrize("endog_names", [None, "target"])
    def test_endog_names(self, endog_names):
        # test that the data has the correct endogenous variable name
        model = ModelBase(
            np.arange(N_POLICIES), np.identity(N_POLICIES), endog_names=endog_names
        )
        if endog_names is None:
            assert model.endog_names == "y"
        else:
            assert model.endog_names == endog_names

    def get_params_cov(self, ols_results):
        params = ols_results.params
        if isinstance(params, pd.Series):
            params = params.values

        cov = ols_results.cov_params()
        if isinstance(cov, pd.DataFrame):
            cov = cov.values

        return params, cov

    def compare_model_to_ols_results(self, model, ols_results):
        # make sure the model's attributes match those of the OLS results
        params, cov = self.get_params_cov(ols_results)
        np.testing.assert_almost_equal(model.mean, params)
        np.testing.assert_almost_equal(model.cov, cov)
        np.testing.assert_array_equal(model.exog_names, ols_results.model.exog_names)
        assert model.endog_names == ols_results.model.endog_names

    def test_from_results(self, ols_results):
        # test that you can initialize a model from statsmodels results object
        model = ModelBase.from_results(ols_results)
        self.compare_model_to_ols_results(model, ols_results)

    def test_to_and_from_csv(self, ols_results):
        # test that you can initialize a model from a csv file
        ModelBase.from_results(ols_results).to_csv(bytes := io.BytesIO())
        bytes.seek(0)
        model = ModelBase.from_csv(bytes)
        self.compare_model_to_ols_results(model, ols_results)


results = ModelBase(np.arange(N_POLICIES), np.identity(N_POLICIES)).fit()


class TestResults:
    def test_conf_int(self):
        with pytest.raises(AttributeError):
            results.conf_int()

    def test_save(self):
        results.save(filename := "temp.p")
        with open(filename, "rb") as results_file:
            loaded_results = pickle.load(results_file)
        os.remove(filename)
        np.testing.assert_almost_equal(loaded_results.model.mean, results.model.mean)

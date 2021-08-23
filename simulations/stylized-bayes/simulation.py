import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, uniform

from conditional_inference.rqu import RQU
from conditional_inference.bayes.classic import LinearClassicBayes
from conditional_inference.bayes.empirical import JamesStein
from conditional_inference.bayes.hierarchical import LinearHierarchicalBayes

RESULTS_DIR = "results"
HYPERPRIOR_LOC = np.linspace(0, .9, num=10)
N_POLICIES = (5, 10, 20)
BETA = (0, 1, 2)
PARAM_SPACE = product(HYPERPRIOR_LOC, N_POLICIES, BETA)


def run_simulation(hyperprior_loc, n_policies, beta):
    # sample true means, sample means, and other parameters
    prior_cov_params_distribution = uniform(hyperprior_loc, .1)
    prior_std = prior_cov_params_distribution.rvs()
    X = np.arange(n_policies).reshape(-1, 1)
    true_mean = multivariate_normal.rvs(
        beta * X.squeeze(), prior_std ** 2 * np.identity(n_policies)
    )
    cov = np.identity(n_policies)
    mean = multivariate_normal.rvs(true_mean, cov)  # sample mean

    # set up estimators
    # estimators maps the estimator name to a tuple of
    # (estimator class, constructor keyword arguments, fit method keyword arguments)
    estimators = {
        "Conventional": (LinearClassicBayes, dict(prior_cov=np.inf), {}),
        "Hybrid": (RQU, {}, dict(beta=0.005)),
        "JamesStein": (JamesStein, {}, {}),
        "FeaturizedJamesStein": (JamesStein, dict(X=X), {}),
        "Hierarchical": (
            LinearHierarchicalBayes,
            dict(prior_cov_params_distribution=prior_cov_params_distribution),
            {},
        ),
        "FeaturizedHierarchical": (
            LinearHierarchicalBayes,
            dict(X=X, prior_cov_params_distribution=prior_cov_params_distribution),
            {},
        ),
    }

    # estimate the models and record point estimates and confidence intervals
    dfs = []
    for name, (model_cls, init_kwargs, fit_kwargs) in estimators.items():
        results = model_cls(mean, cov, **init_kwargs).fit(**fit_kwargs)
        conf_int = results.conf_int()
        df = pd.DataFrame(
            dict(
                estimator=name,
                policy_no=np.arange(n_policies),
                true_mean=true_mean,
                point=results.params,
                ppf_025=conf_int[:, 0],
                ppf_975=conf_int[:, 1]
            )
        )
        if hasattr(results, "posterior_mean_rvs"):
            df["estimated_std"] = results.posterior_mean_rvs.std(axis=1, ddof=1).mean()
        dfs.append(df)
    df = pd.concat(dfs)
    df["true_std"] = true_mean.std(ddof=1)
    
    return df


if __name__ == "__main__":
    sim_no = int(sys.argv[1])
    np.random.seed(sim_no)

    # run simulations for all parameter combinations in the parameter space
    dfs = []
    for hyperprior_loc, n_policies, beta in PARAM_SPACE:
        df = run_simulation(hyperprior_loc, n_policies, beta)
        df["hyperprior_loc"] = hyperprior_loc
        df["n_policies"] = n_policies
        df["beta"] = beta
        dfs.append(df)
    df = pd.concat(dfs)
    df["sim_no"] = sim_no

    if not os.path.exists("results"):
        os.mkdir("results")
    df.to_csv(os.path.join(RESULTS_DIR, f"results_{sim_no}.csv"), index=False)

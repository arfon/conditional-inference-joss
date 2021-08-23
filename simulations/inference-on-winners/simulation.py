import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

from conditional_inference.rqu import RQU

RESULTS_DIR = "results"
N_POLICIES = (2, 10, 50)
BEST_EFFECT = range(10)
PARAM_SPACE = product(N_POLICIES, BEST_EFFECT)

def run_simulation(n_policies, best_effect):
    def get_estimates(name, func):
        point, ppf_025, ppf_975 = func()
        return dict(
            estimator=name,
            point=point,
            ppf_025=ppf_025,
            ppf_975=ppf_975,
        )

    def compute_conventional_estimate():
        return norm.ppf(
            [.5, .025, .975],
            loc=conventional_point_estimate,
            scale=conventional_std
        )

    def compute_conditional_estimate():
        return RQU(mean, cov).get_distribution().ppf([.5, .025, .975])

    def compute_hybrid_estimate():
        beta = .005
        alpha = (.05 - beta) / (1 - beta)
        return RQU(mean, cov).get_distribution(beta=beta)\
            .ppf([.5, alpha / 2, 1 - alpha / 2])

    def compute_projection_estimate():
        c_alpha = RQU(mean, cov).compute_projection_quantile()
        projection = c_alpha * conventional_std
        return np.array(
            [
                conventional_point_estimate,
                conventional_point_estimate - projection,
                conventional_point_estimate + projection
            ]
        )

    estimators = [
        ("Conventional", compute_conventional_estimate),
        ("Conditional", compute_conditional_estimate),
        ("Hybrid", compute_hybrid_estimate),
        ("Projection", compute_projection_estimate)
    ]

    # simulate conventional mean and covariance estimates
    cov = np.identity(n_policies)
    mean = multivariate_normal.rvs([best_effect] + (n_policies - 1) * [0], cov)

    # select best mean and associated variance
    index = int(mean.argmax())
    conventional_point_estimate = mean[index]
    conventional_std = np.sqrt(cov[index, index])

    # compute estimates
    df = pd.DataFrame([get_estimates(name, func) for name, func in estimators])
    df["recommended_effect"] = best_effect if index == 0 else 0
    return df

if __name__ == "__main__":
    sim_no = int(sys.argv[1])
    np.random.seed(sim_no)

    # run simulations for every parameter combination in the parameter space
    dfs = []
    for n_policies, best_effect in PARAM_SPACE:
        df = run_simulation(n_policies, best_effect)
        df["n_policies"] = n_policies
        df["best_effect"] = best_effect
    df = pd.concat(dfs)
    df["sim_no"] = sim_no

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    filename = os.path.join(RESULTS_DIR, f"results_npolicies={n_policies}_besteffect={best_effect}_{sim_no}.csv")
    df.to_csv(filename, index=False)

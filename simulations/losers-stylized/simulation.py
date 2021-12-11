import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal

from conditional_inference.rqu import RQU

RESULTS_DIR = "results"
N_POLICIES  = (3, 11, 51)
BEST_EFFECT = range(10)
WORST_EFFECT = range(0, -10, -1)
PARAM_SPACE = product(N_POLICIES, BEST_EFFECT, WORST_EFFECT)

def run_simulation(n_policies, best_effect, worst_effect):
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
        return RQU(mean, cov).get_distribution(rank=half_n_policies)\
            .ppf([.5, .025, .975])

    def compute_hybrid_estimate():
        beta = .005
        alpha = (.05 - beta) / (1 - beta)
        return RQU(mean, cov).get_distribution(rank=half_n_policies, beta=beta)\
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
    half_n_policies = int((n_policies - 1) / 2)
    true_mean = half_n_policies * [worst_effect] + [0] + half_n_policies * [best_effect]
    mean = multivariate_normal.rvs(true_mean, cov)  # sample mean

    # select kth-best mean and associated variance
    index = mean.argsort()[half_n_policies]
    conventional_point_estimate = mean[index]
    conventional_std = np.sqrt(cov[index, index])

    # compute estimates
    df = pd.DataFrame([get_estimates(name, func) for name, func in estimators])
    df["kth_best_effect"] = true_mean[index]
    return df

if __name__ == "__main__":
    sim_no = int(sys.argv[1])
    np.random.seed(sim_no)

    # run simulations for every parameter combination in the parameter space
    dfs = []
    for n_policies, best_effect, worst_effect in PARAM_SPACE:
        df = run_simulation(n_policies, best_effect, worst_effect)
        df["n_policies"] = n_policies
        df["best_effect"] = best_effect
        df["worst_effect"] = worst_effect
        dfs.append(df)
    df = pd.concat(dfs)
    df["sim_no"] = sim_no

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    filename = os.path.join(RESULTS_DIR, f"results_{sim_no}.csv")
    df.to_csv(filename, index=False)

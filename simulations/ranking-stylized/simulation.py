import os
import sys
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from conditional_inference.rqu import RQU

RESULTS_DIR = "results"
MU_0 = np.linspace(.05, .2, 4)
N_POLICIES = [10, 20, 50]
N_SAMPLES = 10000


def run_simulation(mu_0, n_policies):
    def get_projection_results():
        diff_x = x[index] - np.delete(x, index)
        temp = np.repeat(np.atleast_2d(np.delete(cov[index], index)), len(x) - 1, axis=0)
        diff_cov = cov[index, index] + np.delete(np.delete(cov, index, axis=0), index, axis=1) - temp - temp.T
        results = RQU(diff_x, diff_cov).fit(projection=True)
        return 1 - results.pvalues.max()

    def get_bootstrap_results():
        rvs = multivariate_normal(x, cov).rvs(10000)
        return np.identity(n_policies)[rvs.argmax(axis=1)].mean(axis=0)[index]


    models = {
        "projection": get_projection_results,
        "bootstrap": get_bootstrap_results
    }

    mu = np.insert(np.zeros(n_policies - 1), 0, mu_0)
    cov = np.identity(n_policies)
    x = multivariate_normal(mu, cov).rvs()
    index = x.argmax()

    df = pd.DataFrame(
        [{"model": key, "pr_best": value()} for key, value in models.items()]
    )
    df["mu_0"] = mu_0
    df["n_policies"] = n_policies
    df["selected_best"] = index == 0
    return df


if __name__ == "__main__":
    sim_no = sys.argv[1]
    sim_no = int(sim_no)
    np.random.seed(sim_no)

    df = pd.concat(
        [
            run_simulation(mu_0, n_policies)
            for mu_0, n_policies in product(MU_0, N_POLICIES)
        ]
    )
    df["sim_no"] = sim_no
    df.to_csv(os.path.join(RESULTS_DIR, f"sim_no={sim_no}.csv"), index=False)

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from conditional_inference.bayes.classic import LinearClassicBayes
from conditional_inference.rqu import RQU

RESULTS_DIR = "results"
TOP_N = np.arange(1, 11)
CONVENTIONAL = "conventional"
CONDITIONAL = "conditional"
HYBRID = "hybrid"
PROJECTION = "projection"


def run_simulation(estimator, data_file):
    def compute_results(top_n, results=None, conf_int=None):
        cols = np.arange(top_n)
        if estimator in (CONDITIONAL, HYBRID):
            results = rqu.fit(cols, rank=cols, beta=.005 if estimator == HYBRID else 0)
            conf_int = results.conf_int()
        return {
            "top_n": top_n,
            "rank": cols,
            "true_value": true_mean[cols],
            "params": results.params[cols],  # median estimate
            "ppf025": conf_int[cols, 0],
            "ppf975": conf_int[cols, 1]
        }

    df = pd.read_csv(data_file)
    true_mean, cov = df.values[:, 0], df.values[:, 1:]
    estimated_mean = multivariate_normal(true_mean, cov).rvs()
    argsort = (-estimated_mean).argsort()
    true_mean, estimated_mean, cov = (
        true_mean[argsort], estimated_mean[argsort], cov[argsort][:, argsort]
    )

    results, conf_int = None, None
    if estimator == CONVENTIONAL:
        # cache conventional results
        results = LinearClassicBayes(
            estimated_mean, cov, prior_cov=np.inf
        ).fit(cols=np.arange(TOP_N[-1]))
        conf_int = results.conf_int()
    else:
        rqu = RQU(estimated_mean, cov)
        if estimator == PROJECTION:
            # cache projection results
            results = rqu.fit(cols=np.arange(TOP_N[-1]), projection=True)
            conf_int = results.conf_int()

    return pd.concat(
        [pd.DataFrame(compute_results(top_n, results, conf_int)) for top_n in TOP_N]
    )


if __name__ == "__main__":
    sim_no, estimator, data_file = sys.argv[1:]
    sim_no = int(sim_no)
    data_file_stub = data_file[data_file.rfind("/")+1:-len(".csv")]
    np.random.seed(sim_no)

    df = run_simulation(estimator, data_file)
    df["sim_no"] = sim_no
    df["estimator"] = estimator
    df["data_file"] = data_file_stub

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    filename = os.path.join(
        RESULTS_DIR, f"results_{sim_no}_{estimator}_{data_file_stub}.csv"
    )
    df.to_csv(filename, index=False)

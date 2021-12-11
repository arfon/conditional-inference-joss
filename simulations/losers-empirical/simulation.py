import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from conditional_inference.bayes.classic import LinearClassicBayes
from conditional_inference.rqu import RQU

RESULTS_DIR = "results"
CONVENTIONAL = "conventional"
CONDITIONAL = "conditional"
HYBRID = "hybrid"
PROJECTION = "projection"


def run_simulation(estimator, data_file):
    df = pd.read_csv(data_file)
    true_mean, cov = df.values[:, 0], df.values[:, 1:]
    estimated_mean = multivariate_normal(true_mean, cov).rvs()
    argsort = estimated_mean.argsort()
    true_mean, estimated_mean, cov = (
        true_mean[argsort], estimated_mean[argsort], cov[argsort][:, argsort]
    )
    

    rqu = RQU(estimated_mean, cov)
    estimators = {
        CONVENTIONAL: lambda: LinearClassicBayes(
            estimated_mean, cov, prior_cov=np.inf
        ).fit(),
        CONDITIONAL: lambda: rqu.fit(),
        HYBRID: lambda: rqu.fit(beta=.005),
        PROJECTION: lambda: rqu.fit(projection=True)
    }
    results = estimators[estimator]()
    conf_int = results.conf_int()
    return {
        "rank": np.arange(len(true_mean)),
        "true_value": true_mean,
        "params": results.params,
        "ppf025": conf_int[:, 0],
        "ppf975": conf_int[:, 1]
    }


if __name__ == "__main__":
    sim_no, estimator, data_file = sys.argv[1:]
    sim_no = int(sim_no)
    data_file_stub = data_file[:-len(".csv")]
    np.random.seed(sim_no)

    df = pd.DataFrame(run_simulation(estimator, data_file))
    df["sim_no"] = sim_no
    df["estimator"] = estimator
    df["data_file"] = data_file_stub

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    filename = os.path.join(RESULTS_DIR, f"results_{sim_no}_{estimator}_{data_file_stub}.csv")
    df.to_csv(filename, index=False)

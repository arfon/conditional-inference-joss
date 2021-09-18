import os
import sys

from numpy.random.mtrand import multivariate_normal
import pandas as pd
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from conditional_inference.bayes.classic import LinearClassicBayes
from conditional_inference.rqu import RQU

RESULTS_DIR = "results"
DATA_FILE = "data.csv"

CONVENTIONAL = "conventional"
CONDITIONAL = "conditional"
HYBRID = "hybrid"
PROJECTION = "projection"


def run_simulation(estimator):
    df = pd.read_csv(DATA_FILE)
    true_mean, cov = df.iloc[:, 0].values, df.iloc[:, 1:].values
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
    sim_no, estimator = int(sys.argv[1]), sys.argv[2]
    np.random.seed(sim_no)

    df = pd.DataFrame(run_simulation(estimator))
    df["sim_no"] = sim_no
    df["estimator"] = estimator

    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    filename = os.path.join(RESULTS_DIR, f"results_{estimator}_{sim_no}.csv")
    df.to_csv(filename, index=False)

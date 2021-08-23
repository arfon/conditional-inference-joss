import glob
import logging
import os
import shutil

import pandas as pd

from cluster_setup import OUTPUT_DIR, VARIABLES_TXT, ROOT_DIR_TXT

RESULTS_DIR = "results"
DATA_DIR = "data"
FILENAME = "data.csv"

if __name__ == "__main__":
    # get the directory containing the simulations
    root_dir = open(ROOT_DIR_TXT, "r").read()
    if not os.path.exists(root_dir):
        raise ValueError(f"Directory {root_dir} does not exist.")

    # create a directory to store the results
    data_path = os.path.join(root_dir, DATA_DIR)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # concatenate and store the results
    dfs = []
    results_path = os.path.join(root_dir, RESULTS_DIR)
    for path in glob.glob(os.path.join(results_path, "*.csv")):
        try:
            dfs.append(pd.read_csv(path))
        except:
            logging.warning(f"Error with file {path}")
    pd.concat(dfs).to_csv(os.path.join(data_path, FILENAME), index=False)

    # remove results files that are no longer needed]
    os.remove(VARIABLES_TXT)
    os.remove(ROOT_DIR_TXT)
    shutil.rmtree(results_path)
    shutil.rmtree(OUTPUT_DIR)

# Running simulations

## Setting up the cluster environment

This is a one-time setup.

1. Clone the repository (parent folder)
2. Load your python version e.g., `module load python/python-3.8.5`
3. In the root directory of the repo, create an environment with `python venv venv`
4. Upgrade pip `python -m pip install -U pip`
5. Install the requirements `pip install -r requirements.txt`
6. Install an editable version of the package `pip install -e .`

If you run into an error, check that you're using the pip installed in the virtual environment, not the "gobal" pip:

```
$ which pip
/path/to/conditional-inference/venv/bin/pip
```

Test your installation with:

```python
>>> import conditional_inference
```

## Running a simulation

### Set up

1. Login with `qlogin`
2. Run the cluster setup with `sh cluster_setup.sh <simulation-directory>`
3. Logout `logout`

### Run

4. Submit the simulations to the cluster with `qsub -t 1-$(wc -l < variables.txt) simulation.sh`

### Tear down

5. Login with `qlogin`
6. Tear down the cluster with `sh cluster_teardown.sh`
7. Logout with `logout`
8. Use a file transfer protocol to transfer `<simulation-directory>/data/data.csv` to your local machine. DO NOT commit a giant csv of simulation results to the repo.

## Creating a new simulation folder

A minimal simulations folder will contain the following files:

1. `variables.py` should have an attribute `variables` which is an iterable of command line arguments that will be passed to `simulation.py`
2. `simulation.py` should store (uniquely named) results csv files in a `results` folder
3. A notebook or other analysis file to process the concatenated results stored in `data/data.csv`

## Explanation of tasks

1. `cluster_setup.sh` performs the following tasks:
    1. Reads variables from `<simulation-directory>.variables.variables` and writes them to a file named `variables.txt`
    2. Writes the simulation directory to `root.txt`
    3. Creates a directory to store the cluster output
2. `qsub` performs the following tasks:
    1. Reads the variables from a line of `variables.txt`
    2. Changes into the simulation directory stored in `root.txt`
    3. Runs `simulation.py` with the variables
3. `cluster_teardown.sh` performs the following tasks:
    1. Concatenates the results of the simulations (stored in `<simulation-directory>/results)` and stores them in `<simulation-directory>/data/data.csv`
    2. Removes results files and folders that are no longer needed.

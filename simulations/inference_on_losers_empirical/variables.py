from itertools import product

from .simulation import CONVENTIONAL, CONDITIONAL, HYBRID, PROJECTION

N_SIMULATIONS = 1000

estimators = [CONVENTIONAL, CONDITIONAL, HYBRID, PROJECTION]
variables = product(range(N_SIMULATIONS), estimators)

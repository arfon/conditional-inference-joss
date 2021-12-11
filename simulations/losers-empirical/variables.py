from itertools import product

from .simulation import CONVENTIONAL, CONDITIONAL, HYBRID, PROJECTION

N_SIMULATIONS = 5000

estimators = [CONVENTIONAL, CONDITIONAL, HYBRID, PROJECTION]
variables = product(range(N_SIMULATIONS), estimators, ("movers.csv", "oa.csv"))

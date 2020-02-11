import numpy as np
from sampling import run

run_smc = run.RunSMC

def constraints(x):
    return np.array([x[0] - np.sqrt(14*(x[1]**2) + 2), -(x[0] -np.sqrt(33*(x[1]**2) + 1))])

a = np.array([0.2, 0.4])
# bounds = np.array([[0.4, 0.6], [0.1, 0.47], [0.1, 0.47], [0.03, 0.08]])
bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
print(constraints(a), bounds)

run_smc(N=1000, bounds=bounds, scmc_type="constrained_scmc", tau_T=10e6, constraints=constraints)
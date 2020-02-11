import numpy as np
from sampling import run

run_scmc = run.RunSCMC

def constraints(x):
    return [x[0] - np.sqrt(14*(x[1]**2) + 2), -(x[0] -np.sqrt(33*(x[1]**2) + 1))]

a = np.array([0.2, 0.4])
print(constraints(a))
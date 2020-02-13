####################Python packages#####################
import argparse
import numpy as np

####################Custom imports#####################
import matplotlib.pyplot as plt
from sampling import run

run_smc = run.RunSMC

def constraints(x):
    return -np.array([x[0] - np.sqrt(14*(x[1]**2) + 2), -(x[0] -np.sqrt(33*(x[1]**2) + 1))])

a = np.array([0.2, 0.4])
# bounds = np.array([[0.4, 0.6], [0.1, 0.47], [0.1, 0.47], [0.03, 0.08]])
bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
print(constraints(a), bounds)

run = run_smc(N=1000, bounds=bounds, scmc_type="constrained_scmc", tau_T=1e6, constraints=constraints)
x, x0 = run.x, run.x0

print(np.sum(np.array([constraints(_x)[0]<=0 and constraints(_x)[1]<=0 for _x in x]))/1000.0)
plt.plot(x0[:,0], x0[:,1], 'o', x[:,0],x[:,1],'*')
plt.show()
# 
# if __name__ == 'main':
#     parser = argparse.ArgumentParser(description='Process inputs to sampler')
#     parser.add
#     
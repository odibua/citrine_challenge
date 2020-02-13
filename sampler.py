####################Python packages#####################
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as path

####################Custom imports#####################
from constraints import Constraint
from sampling.run import RunSMC
from utils import get_accuracy, get_bounds

# def func_constraints(x):
#     return np.array([x[0] - np.sqrt(14*(x[1]**2) + 2), -(x[0] -np.sqrt(33*(x[1]**2) + 1))])

# a = np.array([0.2, 0.4])
# bounds = np.array([[-2.0, 2.0], [-2.0, 2.0]])
# print(constraints(a), bounds)
#
# print(np.sum(np.array([constraints(_x)[0]<=0 and constraints(_x)[1]<=0 for _x in x]))/1000.0)
# plt.plot(x0[:, 0], x0[:, 1], 'o', x[:, 0], x[:, 1], '*')
# plt.show()


def main():
    parser = argparse.ArgumentParser(description='Process inputs to sampler')
    parser.add_argument('input_file', default=None, type=str)
    parser.add_argument('output_file', default="output.txt", type=str)
    parser.add_argument('N', default=1000, type=int)

    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    N = args.N
    print(input_file, output_file, N)

    if not path.isfile(input_file) and input_file:
        raise ValueError('Input file does not exist')
    elif input_file:
        constraint = Constraint(input_file)
        x0 = np.array(constraint.get_example())
        constraints = constraint.eval_constraints
        constraint_bool = constraint.apply_list
    else:
        constraint = func_constraints
        constraint_bool = func_constraints_bool
        x0 = x0

    bounds = get_bounds(x0, constraint_bool)
    run = RunSMC(N=N, bounds=bounds, type="constrained_scmc", tau_T=1e6, constraints=constraints)
    x, x0 = run.get_x(), run.get_x0()
    import ipdb
    ipdb.set_trace()
    np.savetxt(output_file, x, delimiter=' ')
    np.savetxt(output_file+'0', x0, delimiter=' ')


if __name__ == "__main__":
    main()

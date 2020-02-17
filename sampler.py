####################Python packages#####################
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os.path as path

####################Custom imports#####################
from constraints import Constraint
from sampling.run import RunSMC
from utils import get_accuracy, get_bounds


def main():
    # Obtain input arguments
    parser = argparse.ArgumentParser(description='Process inputs to sampler')
    parser.add_argument('input_file', default=None, type=str)
    parser.add_argument('output_file', default="output.txt", type=str)
    parser.add_argument('N', default=1000, type=int)
    parser.add_argument('--plot_bool', action="store_true")

    # Parse input arguments
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file
    N = args.N
    plot_bool = args.plot_bool

    np.random.seed(0)

    # Load file and the corresponding constraint functions and valid example
    if not path.isfile(input_file) and input_file:
        raise ValueError('Input file does not exist')
    elif input_file:
        constraint = Constraint(input_file)
        x0 = np.array(constraint.get_example())
        constraints = constraint.eval_constraints
        constraint_bool = constraint.apply
        constraint_bools = constraint.apply_list
    else:
        # Used for local testing
        constraints = func_constraints
        constraint_bool = func_constraints_bool
        x0 = x0

    # Obtain bounds of input space from which initial sample guesses will
    # be uniformly drawn
    bounds = get_bounds(x0, constraint_bools)

    # Find valid uniformly distributed samples with constrained SMC
    run = RunSMC(N=N, bounds=bounds, type="constrained_smc", tau_T=1e7, constraints=constraints)
    x, x0 = run.get_x(), run.get_x0()

    # Evaluate validity. uniqueness, and spread of results
    acc = get_accuracy(x, constraint_bool)
    std = np.std(x, axis=0)
    uniq_nums = np.unique(x, axis=0).shape[0]

    print('accuracy: {acc} std: {std} unique candidates: {uniq}'.format(acc=acc, std=std, uniq=uniq_nums))

    # Save output and plot if 2D and plot_bool
    np.savetxt(output_file, x, delimiter=' ')
    if x.shape[1] == 2 and plot_bool:
        plt.plot(x0[:, 0], x0[:, 1], 'o', x[:, 0], x[:, 1], '*')
        plt.show()


if __name__ == "__main__":
    main()

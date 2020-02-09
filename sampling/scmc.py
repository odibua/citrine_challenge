import numpy as np
from scipy.stats import norm
from smt.sampling_methods import LHS
from typing import List, Callable


class constrained_scmc:
    def __init__(self, N: int, bounds: np.Array[float], constraints: List[Callable], tau_T: float):
        """
        Initialize scmc class with relevant parameters and constraints
        :param N: Number of samples to be generated
        :param bounds: Lower and upper bounds of hypercube in each dimension
        :param constraints: List of constraint evaluations
        """
        # Initialize constraints and weights
        self.N = N
        self.constraints = constraints
        self.W = np.array([1.0/N]*N)
        self.w = np.array([0.0]*N)

        # Initialize candidate points
        sampling = LHS(xlimits=bounds)
        self.x = sampling(N)

        # Save initial constant and goal constant to state
        self.tau_T = tau_T

        # Initialize normal cdf with variance of 1
        self.norm_cdf = norm.cdf
        self.scale = 1.0

    def modify_weights(self, tau_t: float):
        """
        Modify the weights at time step t based on constraints and current candidate
        :param tau:
        :return:
        """
        pass


    def calc_wn(self, x: np.Array[float], tau_t: float, tau_t_1: float, constraints: List[Callable]):
        """
        Calculate wn which is used to modify weights for resampling, and adaptively determine tau_t
        :param x: Array of sampled points for which wns will be calculated
        :param tau_t: Current value of tau
        :param tau_t_1:  Previous value of tau
        :param constraints: List of constraint evaluations
        """
        for idx, _x in enumerate(x):
            num = np.prod(self.norm_cdf(-tau_t*constraints))
            den = np.prod(self.norm_cdf(-tau_t_1*constraints.eval_constraints()))
            self.w[idx] = num/den















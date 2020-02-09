import numpy as np
from random import choices
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
        self.W = self.init_weights(N)
        self.w = np.array([0.0]*N)

        # Initialize candidate points
        sampling = LHS(xlimits=bounds)
        self.x = sampling(N)

        # Save initial constant and goal constant to state
        self.tau_T = tau_T

        # Initialize normal cdf with variance of 1
        self.norm_cdf = norm.cdf
        self.scale = 1.0


    def modify_weights(self, tau_t_1: float, tau_t: float):
        """
        Modify the weights at time step t based on constraints and current candidate
        :param tau_t: Current value of tau
        :param tau_t_1:  Previous value of tau
        :return:
        """

        # Calculate wns used to modify weights
        self.calc_wn(tau_t_1, tau_t)

        #M Modify weights
        norm_constant = 0
        for idx, W in enumerate(self.W):
            self.W[idx] = W*self.w[idx]
            norm_constant = norm_constant + self.W[idx]
        self.W = self.W/norm_constant

    def calc_wn(self, tau_t_1: float, tau_t: float):
        """
        Calculate wn which is used to modify weights for resampling, and adaptively determine tau_t
        :param tau_t: Current value of tau
        :param tau_t_1:  Previous value of tau
        """
        for idx, _x in enumerate(self.x):
            num = np.prod(self.norm_cdf(-tau_t*self.constraints))
            den = np.prod(self.norm_cdf(-tau_t_1*self.constraints.eval_constraints(_x)))
            self.w[idx] = num/den

    # TODO: Make part of run_scmc class
    def resample_candidates(self):
        """
        Resample from candidate points based on updated weights
        :return:
        """
        self.x = np.array(choices(population=self.x, weights=self.W, k=self.N))

    @staticmethod
    def init_weights(N: int) -> np.Array[float]:
        """
        Initialize weights to be uniform in number of candidae points
        :param N: Number of candidate poins
        :return:
        """
        return np.array([[1.0/N]*N])


















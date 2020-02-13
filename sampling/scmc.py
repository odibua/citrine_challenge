####################Python packages#####################
import numpy as np
from random import choices
from scipy import optimize
from scipy.stats import norm
from smt.sampling_methods import LHS
from typing import Callable, List


class ConstrainedSCMC:
    def __init__(self, N: int, bounds: np.ndarray, constraints: List[Callable], tau_T: float):
        """
        Initialize scmc class with relevant parameters and constraints
        :param N: Number of samples to be generated
        :param bounds: Lower and upper bounds of hypercube in each dimension
        :param constraints: List of constraint that define valid regions of the hypercube
        """
        # Initialize constraints and weights
        self.N = N
        self.constraints = constraints
        self.init_weights(N)
        self.w = np.array([0.0]*N)

        # Initialize candidate points
        sampling = LHS(xlimits=bounds)
        self.x = sampling(N)

        # Save initial constant and goal constant to state
        self.tau_t = 0
        self.tau_T = tau_T

        # Initialize normal cdf with variance of 1
        self.norm_cdf = norm.cdf
        self.scale = 1.0

        # Initialize desired Effective Sample Size
        self.ess = 0.5*N

    def modify_weights(self):
        """
        Modify the weights at time step t based on constraints and current candidate
        :param tau_t: Current value of tau
        :param tau_t_1:  Previous value of tau
        :return:
        """
        #

        # Calculate wns used to modify weights
        calc_wn = self.outer_calc_wn(self.norm_cdf, self.x, self.w, self.constraints, self.tau_t, self.scale)
        self.tau_t =  self.get_tau_t(calc_wn=calc_wn, tau_t=self.tau_t+1e-3, ess=self.ess)
        self.w = calc_wn(self.tau_t)

        #M Modify weights
        norm_constant = 0
        for idx, W in enumerate(self.W):
            self.W[idx] = W*self.w[idx]
            norm_constant = norm_constant + self.W[idx]
        self.W = self.W/norm_constant

    @staticmethod
    def outer_calc_wn(norm_cdf: Callable, x: np.ndarray, w: np.ndarray, constraints: List[Callable], tau_t_1: float, scale: float = 1.0):
        """
        Calculate wn which is used to modify weights for resampling, and adaptively determine tau_t
        :param norm_cdf: CDF of normal distribution function
        :param x: Array of x candidate points
        :param constraints: List of constraint evaluations
        :param tau_t_1:  Previous value of tau
        :return calc_wn function that takes as an argument tau_t
        """
        def calc_wn(tau_t):
            for idx, _x in enumerate(x):
                num = np.sum(np.log(norm_cdf(tau_t*constraints(_x), scale=scale)))
                den = np.sum(np.log(norm_cdf(tau_t_1*constraints(_x), scale=scale)))
                w[idx] = num - den
            return np.exp(w)
        return calc_wn

    # TODO: Make part of run_scmc class
    def resample_candidates(self):
        """
        Resample from candidate points based on updated weights
        :return:
        """
        # import ipdb
        # ipdb.set_trace()
        self.x = np.array(choices(population=self.x, weights=self.W, k=self.N))

    def stop(self) -> int:
        """
        States whether stopping condition of tau_t>=tau_T has been reached
        :return: Booleam
        """
        if self.tau_t>=self.tau_T:
            return True
        return False

    def init_weights(self, N: int) -> np.ndarray:
        """
        Initialize weights to be uniform in number of candidae points
        :param N: Number of candidate poins
        :return:
        """
        self.W = np.array([1.0/N]*N)

    @staticmethod
    def get_tau_t(calc_wn: Callable, tau_t: float, ess: float):
        # F = lambda tau_t: np.exp(np.log(sum(calc_wn(tau_t))**2) - np.log(sum(calc_wn(tau_t)**2))) - ess
        F = lambda tau_t: sum(calc_wn(tau_t)) ** 2/(sum(calc_wn(tau_t) ** 2)) - ess
        tau = optimize.fsolve(F, tau_t)
        return tau

    def get_pi(self) -> Callable:
        def target_distrib(tau_t: float, constraints: List[Callable], scale: float, norm_cdf: Callable) -> Callable:
            def pi(x: np.ndarray) -> float:
                return np.exp(np.sum(np.log(norm_cdf(tau_t * constraints(x), scale=scale))))
            return pi
        return target_distrib(self.tau_t, self.constraints, scale=self.scale, norm_cdf=self.norm_cdf)


















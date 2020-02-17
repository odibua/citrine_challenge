####################Python packages#####################
import numpy as np
from random import choices
from typing import Callable, List

##################Third Party Packages###################
from numba import jit
from scipy import optimize
from scipy.stats import norm
from smt.sampling_methods import LHS


class ConstrainedSCMC:
    def __init__(self, N: int, bounds: np.ndarray, constraints: List[Callable], tau_T: float):
        """
        Initialize scmc class with relevant parameters and constraints. Implemented as
        detailed in S. Golchi et al, "Monte Carlo based Designs for Constrained Domains" 
        (https://arxiv.org/pdf/1512.07328.pdf)
        :param N: Number of samples to be generated
        :param bounds: Lower and upper bounds of hypercube in each dimension
        :param constraints: List of constraint that define valid regions of the hypercube
        :param tau_T: Threshold value of constraint coefficient
        """
        # Initialize constraints and weights
        self.N = N
        self.constraints = constraints
        self.init_weights(N)
        self.w = np.array([0.0]*N)

        # Initialize candidate points
        sampling = LHS(xlimits=bounds)
        self.x = sampling(N)
        # import ipdb
        # ipdb.set_trace()
        # Save initial constant and goal constant to state
        self.tau_t = 1e-1
        self.tau_T = tau_T

        # Initialize normal cdf with variance of 1
        self.norm_cdf = norm.cdf
        self.scale = 1.0

        # Initialize desired Effective Sample Size
        self.ess = 0.5*N

    def modify_weights(self):
        """
        Modify the weights based on constraints and current candidate
        """
        #

        # Calculate wns used to modify weights
        calc_wn = self.outer_calc_wn(self.norm_cdf, self.x, self.constraints, self.tau_t, self.scale)
        self.tau_t = self.get_tau_t(calc_wn=calc_wn, tau_t=self.tau_t, tau_T=self.tau_T, ess=self.ess)
        self.w = calc_wn(self.tau_t)

        #M Modify weights
        norm_constant = 0
        for idx, W in enumerate(self.W):
            self.W[idx] = W*self.w[idx]
            norm_constant = norm_constant + self.W[idx]
        self.W = self.W/norm_constant

    @staticmethod
    def outer_calc_wn(norm_cdf: Callable, x: np.ndarray, constraints: List[Callable], tau_t_1: float, scale: float = 1.0):
        """
        Calculate wn which is used to modify weights for resampling, and adaptively determine tau_t
        :param norm_cdf: CDF of normal distribution function
        :param x: Array of x candidate points
        :param constraints: List of constraint evaluations
        :param tau_t_1:  Previous value of tau
        :return calc_wn function that takes as an argument tau_t
        """
        def calc_wn(tau_t):
            def _func(_x):
                num = np.sum(np.log(norm_cdf(-tau_t * constraints(_x), scale=scale)))
                den = np.sum(np.log(norm_cdf(-tau_t_1 * constraints(_x), scale=scale)))
                return num - den
            w = np.array(list(map(_func, x)))
            return np.exp(w)
        return calc_wn

    def resample_candidates(self):
        """
        Resample from candidate points based on updated weights
        """
        self.x = np.array(choices(population=self.x, weights=self.W, k=self.N))

    def stop(self) -> int:
        """
        States whether stopping condition of tau_t>=tau_T has been reached
        :return: Boolean
        """
        if np.abs(np.log10(self.tau_t)-np.log10(self.tau_T))<=1e-1:
            return True
        return False

    def init_weights(self, N: int):
        """
        Initialize weights to be uniform in number of candidae points
        :param N: Number of candidate poins
        """
        self.W = np.array([1.0/N]*N)

    @staticmethod
    def get_tau_t(calc_wn: Callable, tau_t: float, tau_T: float, ess: float) -> float:
        """
        Solve optimization problem that yields the constraint coefficient, tau_t for the next
        time step. Optimized in log-scale

        :param calc_wn: Function that calculates wn (used to modify weights)
        :param tau_t: Initial guess of contraint coefficient
        :param tau_T: Largest allowable value of constraint coefficient
        :param ess: Effective sample size tau_t is chosen to optimize for
        :return: tau: Optimal tau_t
        """
        F = lambda tau_tt: sum(calc_wn(10**tau_tt)) ** 2/(sum(calc_wn(10**tau_tt) ** 2)) - ess
        tau = optimize.fminbound(F, np.log10(tau_t), np.log10(tau_T))
        return 10**tau

    def get_pi(self) -> Callable:
        """
        Returns target distribution

        :return: Target distribution
        """
        def target_distrib(tau_t: float, constraints: List[Callable], scale: float, norm_cdf: Callable) -> Callable:
            def pi(x: np.ndarray) -> float:
                return np.exp(np.sum(np.log(norm_cdf(-tau_t * constraints(x), scale=scale))))
            return pi
        return target_distrib(self.tau_t, self.constraints, scale=1.0, norm_cdf=self.norm_cdf)

    def get_x(self):
        return self.x


















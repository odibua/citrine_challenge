####################Python packages#####################
import copy
import numpy as np
from scipy.stats import norm, uniform
from typing import Callable

norm_rvs = norm.rvs
norm_pdf = norm.pdf

class MetropolisHastings:
    def __init__(self, x: np.ndarray, pi: Callable,  T: int = 1, proposal_distrib: Callable = norm_pdf, trans_kern: Callable = norm_rvs, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize basic parameters used for metropolis hastings
        :param x: Initial vector from which new samples will be generated
        :param pi: Function that represents target distribution
        :param T: Number of steps taking in metropolis hastings
        :param trans_kern: Transition kernel used to generate new samples. Defaults to sampling N(0,1)
        :param loc: Mean of transition kernel
        :param scale: Standard deviation of transition kernel
        """

        self. x = [x] if len(x.shape)==1 else x
        self.pi = pi
        self.trans_kern = trans_kern
        self.T = T
        self.loc = loc
        self.scale = scale

        def proposal(xi, xi_1):
            return proposal_distrib(xi-xi_1, loc=self.loc, scale=self.scale)
        self.proposal_distrib = proposal

    def gibbs_type(self, proposal_distrib: Callable = None, trans_kern: Callable = None):
        """
        Perform metropolis hastings using one element of each candidate at a time.
        :param proposal_distrib: Proposal distribution that evaluates pdf of proposed point conditional on new one
        :param trans_kern:  Transition kernel that generates new proposals
        :return: Updated Samples
        """
        self.proposal_distrib = proposal_distrib if proposal_distrib else self.proposal_distrib
        self.trans_kern = trans_kern if trans_kern else self.trans_kern
        # import ipdb
        # ipdb.set_trace()
        for t in range(self.T):
            for idx, _x in enumerate(self.x):
                for idx2, _xi in enumerate(_x):
                    _xi_0, _x_temp = _xi, copy.deepcopy(_x)
                    proposed_xi = _xi_0 + self.trans_kern(loc=self.loc, scale=self.scale)
                    _x_temp[idx2] = proposed_xi

                    num = self.pi(_x_temp)*self.proposal_distrib(_xi_0, proposed_xi)
                    den = self.pi(_x)*self.proposal_distrib(proposed_xi, _xi_0)
                    A = min(1, num/den)
                    if uniform.rvs() < A:
                        _x[idx2] = proposed_xi
                self.x[idx] = _x

        return self.x






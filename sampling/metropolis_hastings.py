####################Python packages#####################
import copy
import numpy as np
from typing import Callable

##################Third Party Packages###################
from scipy.stats import norm, uniform

norm_rvs = norm.rvs
norm_pdf = norm.pdf

class MetropolisHastings:
    def __init__(self, x: np.ndarray, pi: Callable,  T: int = 10, proposal_distrib: Callable = norm_pdf, trans_kern: Callable = norm_rvs, loc: float = 0.0, scale: float = 1.0):
        """
        Initialize basic parameters used for metropolis hastings
        
        :param x: Initial vector from which new samples will be generated
        :param pi: Function that represents target distribution
        :param T: Number of steps taken in metropolis hastings
        :param proposal_distrib: Distribution that gives pdf of x_t conditional on x_t_1
        :param trans_kern: Transition kernel used to generate new samples. Defaults to sampling N(0,1)
        :param loc: Mean of transition kernel
        :param scale: Standard deviation of transition kernel
        """
        self. x = [x] if len(x.shape) == 1 else x
        self.pi = pi
        self.trans_kern = trans_kern
        self.T = T
        self.loc = loc
        self.scale = scale

        def proposal(xi, xi_1, _scale=self.scale):
            return proposal_distrib(xi-xi_1, loc=self.loc, scale=_scale)
        self.proposal_distrib = proposal

    def gibbs_type(self, proposal_distrib: Callable = None, trans_kern: Callable = None):
        """
        Perform metropolis hastings using one element of each candidate at a time.
        
        :param proposal_distrib: Proposal distribution that evaluates pdf of proposed point conditional on new one
        :param trans_kern:  Transition kernel that generates new proposals
        :return: Updated Samples
        """
        scale = self.scale
        self.proposal_distrib = proposal_distrib if proposal_distrib else self.proposal_distrib
        self.trans_kern = trans_kern if trans_kern else self.trans_kern

        def _func(_x, _trans_kern=self.trans_kern, _proposal_distrib=self.proposal_distrib, _loc=self.loc, _scale=scale):
            for idx, _xi in enumerate(_x):
                _xi_0, _x_temp = _xi, copy.deepcopy(_x)
                proposed_xi = _xi_0 + _trans_kern(loc=_loc, scale=_scale[idx])
                _x_temp[idx] = proposed_xi

                num = self.pi(_x_temp) * _proposal_distrib(_xi_0, proposed_xi, _scale=_scale[idx])
                den = self.pi(_x) * _proposal_distrib(proposed_xi, _xi_0, _scale=_scale[idx])
                ratio = np.exp(np.log(num) - np.log(den))
                a = min(1, ratio)
                if uniform.rvs() < a:
                    _x[idx] = proposed_xi
            return _x

        for _ in range(self.T):
            self.x = np.array(list(map(_func, self.x)))

        return self.x






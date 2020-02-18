####################Python packages#####################
import numpy as np
from typing import Callable, List
import copy

##################Third Party Packages###################
from sampling.metropolis_hastings import MetropolisHastings
from sampling.smc import ConstrainedSMC


class RunSMC:
    def __init__(self, N: int, bounds: np.ndarray, type: str, tau_T: float = None, constraints: List[Callable] = None, t: int = 10):
        """
        Run SMC sampling methods based on user input
        
        :param N: Number of points to be sampled
        :param bounds: Bounds of sampling region
        :param type: Type of SMC to be run
        :param tau_T: Stopping condition for constrained SMC
        :param constraints: List constraints that define valid regions of space
        """
        self.t = t
        self.x0, self.x, self.scale = None, None, None
        types = ["constrained_smc"]
        if type not in types:
            raise ValueError('Invalid SMC type try one of {t}.'.format(t=types))

        if type == "constrained_smc":
            if not constraints:
                raise ValueError('No constraints specified for Constrained SMC. Specify constraints as list of functions')
            if not tau_T:
                raise ValueError('No constraint parameter value tau_T specified')
            self.run_SMC(N, bounds, constraints, tau_T)

    def run_SMC(self, N: int, bounds: np.ndarray, constraints: Callable, tau_T: float):
        """
        Run SMC for given bounds and constraints
        
        :param N: Number of samples taken
        :param bounds: Boundary of domain to be sampled
        :param constraints: Function that returns values of constraints
        :param tau_T: Stopping value of tau_t
        :return:
        """
        smc = ConstrainedSMC(N, bounds, constraints, tau_T)
        t = 0
        self.x0 = copy.deepcopy(smc.x)
        self.scale = np.std(self.x0, axis=0)
        while not smc.stop() and t < self.t:
            smc.modify_weights()
            smc.resample_candidates()
            smc.init_weights(N)
            x, pi = smc.get_x(), smc.get_pi()
            mh = MetropolisHastings(x, pi, scale=self.scale)
            smc.x = mh.gibbs_type()
            t = t+1
        self.x = smc.x

    def get_x(self):
        return self.x

    def get_x0(self):
        return self.x0





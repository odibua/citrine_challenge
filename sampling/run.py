####################Python packages#####################
import numpy as np
from typing import Callable, List
import copy

##################Third Party Packages###################
from sampling.metropolis_hastings import MetropolisHastings
from sampling.scmc import ConstrainedSCMC

class RunSMC:
    def __init__(self, N: int, bounds: np.ndarray, type: str, tau_T: float = None, constraints: List[Callable] = None, t: int = 20):
        """
        Run SCMC sampling methods based on user input
        :param N: Number of points to be sampled
        :param bounds: Bounds of sampling region
        :param type: Type of SCMC to be run
        :param tau_T: Stopping condition for constrained scmc
        :param constraints: List constraints that define valid regions of space
        """
        self.t = t
        self.x0, self.x = None, None
        types = ["constrained_scmc"]
        if type not in types:
            raise ValueError('Invalid SCMC type try one of {t}.'.format(t=types))

        if type == "constrained_scmc":
            if not constraints:
                raise ValueError('No constraints specified for Constrained SCMC. Specify constraints as list of functions')
            if not tau_T:
                raise ValueError('No constraint parameter value tau_T specified')
            self.run_scmc(N, bounds, constraints, tau_T)

    def run_scmc(self, N: int, bounds: np.ndarray, constraints: Callable, tau_T: float):
        """
        Run SCMC for given bounds and constraints
        :param N: Number of samples taken
        :param bounds: Boundary of domain to be sampled
        :param constraints: Function that returns values of constraints
        :param tau_T: Stopping value of tau_t
        :return:
        """
        scmc = ConstrainedSCMC(N, bounds, constraints, tau_T)
        t = 0
        self.x0 = copy.deepcopy(scmc.x)
        while not scmc.stop() and t < self.t:
            print(t)
            print(scmc.tau_t)
            scmc.modify_weights()
            scmc.resample_candidates()
            scmc.init_weights(N)
            x, pi = scmc.get_x(), scmc.get_pi()
            mh = MetropolisHastings(x, pi)
            scmc.x = mh.gibbs_type()
            t = t+1
        self.x = scmc.x

    def get_x(self):
        return self.x

    def get_x0(self):
        return self.x0





from sampling.metropolis_hastings import MetropolisHastings
from sampling.scmc import ConstrainedSCMC
import numpy as np
from typing import Callable, List
import copy

class RunSMC:
    def __init__(self, N: int, bounds: np.ndarray, scmc_type: str, tau_T: float = None, constraints: List[Callable] = None):
        """
        Run SCMC sampling methods based on user input
        :param N: Number of points to be sampled
        :param bounds: Bounds of sampling region
        :param type: Type of SCMC to be run
        :param tau_T: Stopping condition for constrained scmc
        :param constraints: List constraints that define valid regions of space
        """

        types = ["constrained_scmc"]
        if scmc_type not in types:
            raise ValueError('Invalid SCMC type try one of {t}.'.format(t=types))

        if scmc_type=="constrained_scmc":
            if not constraints:
                raise ValueError('No constraints specified for Constrained SCMC. Specify constraints as list of functions')
            if not tau_T:
                raise ValueError('No constraint parameter value tau_T specified')
            self.run_scmc(N, bounds, constraints, tau_T)

    def run_scmc(self, N, bounds, constraints, tau_T):
        scmc = ConstrainedSCMC(N, bounds, constraints, tau_T)
        t=0
        self.x0 = copy.deepcopy(scmc.x)
        while not scmc.stop() and t < 10:
            print(t)
            print(scmc.tau_t)
            # import ipdb
            # ipdb.set_trace()
            scmc.modify_weights()
            scmc.resample_candidates()
            scmc.init_weights(N)
            x, pi = scmc.x, scmc.get_pi()
            mh = MetropolisHastings(x, pi)
            scmc.x = mh.gibbs_type()
            t=t+1

        self.x = scmc.x





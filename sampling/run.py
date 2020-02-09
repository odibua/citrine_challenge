from sampling.metropolis_hastings import metropolis_hastings
from sampling.scmc import constrained_scmc
from typing import Callable, List

class run_scmc:
    def __init__(self, N: int, bounds: np.Array[float], type: str, tau_T: float = None, constraints: List[Callable] = None):
        """
        Run SCMC sampling methods based on user input
        :param N: Number of points to be sampled
        :param bounds: Bounds of sampling region
        :param type: Type of SCMC to be run
        :param tau_T: Stopping condition for constrained scmc
        :param constraints: List constraints that define valid regions of space
        """

        types = ["constrained_scmc"]
        if type not in types:
            raise ValueError('Invalid SCMC type try one of {t}.'.format(t=types))

        if type=="constrained_scmc":
            if not constraints:
                raise ValueError('No constraints specified for Constrained SCMC. Specify constraints as list of functions')
            scmc = constrained_scmc(N, bounds, constraints, tau_T)

        while not scmc.stop():
            scmc.modify_weights()
            scmc.resample_candidates()
            scmc.init_weights()
            x, pi = scmc.x, scmc.get_pi()
            mh = metropolis_hastings(x, pi)
            scmc.x = mh.gibbs_type()

        return scmc.x





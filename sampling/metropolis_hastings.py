import numpy as np
from scipy.stats import norm
from typing import Callable

norm_rvs = norm.rvs
class metropolis_hastings:
    def __init__(self, x: np.Array[float], trans_kern: Callable = norm_rvs,  T: int = 5):
        """
        Initialize basic parameters used for metropolis hastings
        :param x: Initial vector from which new samples will be generated
        :param trans_kern: Transition kernel used to generate new samples. Defaults to N(0,1)
        :param T: Number of steps taking in metropolis hastings
        """
        self. x = x
        self.trans_kern = trans_kern
        self.T = T



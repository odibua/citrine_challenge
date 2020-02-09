import numpy as np
from smt.sampling_methods import LHS
from typing import List


class run_scmc:
    def __init__(self, N: int, bounds: np.Array, constraints: List = None):
        """
        Initialize scmc class with relevant parameters and constraints
        :param N: Number of samples to be generated
        :param x0: Initial valid point in constrained domain
        :param constraints: List of constraint evaluations
        """
        # Initialize constraints and weights
        self.N = N
        self.constraints = constraints
        self.W = np.array([1.0/N]*N)

        # Initialize candidate points
        sampling = LHS(xlimits=bounds)
        self.x = sampling(N)




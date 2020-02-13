####################Python packages#####################
import numpy as np
from typing import Callable, List


def get_accuracy(x: List[np.ndarray], constraints: Callable) -> float:
    """
    Return portion of sampled vectors that laid in the constraint
    :param x: List of proposed vectors
    :param constraints: Boolean that states if all constraints are satisfied
    :return: accuracy
    """
    n, n_valid = x0.shape[0], 0.0

    for _x in x:
        n_valid = n_valid + constraints(_x)
    return n_valid / n


def get_bounds(x0: np.ndarray, constraint_bool: Callable, scale: int = 2) -> np.ndarray:
    """
    Returns bounds that violate all of the constraints to insure that we are sampling
    from a domain that contains the full constrained region
    :param x0: x that is in constrained region
    :param constraint_bool: Returns list of booleans that states if vector violates particular constraint
    :return: Bounds that violate all constraints
    """
    bound_violated = False
    bound = np.zeros((x0.shape[0], 2))
    max_val = max(max(np.abs(x0 - scale*x0)), max(np.abs(x0 + scale*x0)))
    bound[:, 0], bound[:, 1] = -max_val, max_val

    while not bound_violated:
        low_bool, up_bool = constraint_bool(bound[:,0]), constraint_bool(bound[:,1])
        if sum(low_bool) == len(low_bool) and sum(up_bool) == len(up_bool):
            return bound

        bound[:, 0] = bound[:, 0] - scale*max_val
        bound[:, 1] = bound[:, 1] + scale*max_val


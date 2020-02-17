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
    n, n_valid = x.shape[0], 0.0
    x_valid = []
    for _x in x:
        valid = constraints(_x)
        n_valid = n_valid + int(valid)
        if valid:
            x_valid.append(_x)
    return n_valid / n, np.array(x_valid)


def get_bounds(x0: np.ndarray, constraint_bool: Callable, scale: int = 1e0, thresh_ratio: float = 1.25) -> np.ndarray:
    """
    Returns bounds that violate all of the constraints to insure that we are sampling
    from a domain that contains the full constrained region
    :param x0: Sample candidate that is in constrained region
    :param constraint_bool: Returns list of booleans that states if vector violates particular constraint
    :param scale: Parameter that controls magnitude of bounds
    :param thresh_ratio: Controls how many constraints should be violated before stopping the growth of bounds
    :return: Bounds that violate all constraints
    """
    # Initialize upper and lower bounds
    bound_violated = False
    bound = np.zeros((x0.shape[0], 2))
    x0 = np.array([x0_i if x0_i != 0 else 0.1 for x0_i in x0])
    bound[:, 0], bound[:, 1] = x0, x0

    # Grow bounds until enough constraints are violated
    while not bound_violated:
        low_bool, up_bool = constraint_bool(bound[:, 0]), constraint_bool(bound[:, 1])
        if len(low_bool) > 1 and sum(low_bool) <= len(low_bool) / thresh_ratio and sum(up_bool) <= len(up_bool) / thresh_ratio:
            return bound
        elif len(low_bool) == 1 and sum(low_bool) + sum(up_bool) <= (len(low_bool) + len(up_bool)) / thresh_ratio:
            return bound

        # Only grow upper/lower bounds while certain number of thresholds are not violated
        if sum(low_bool) > len(low_bool) / thresh_ratio:
            bound[:, 0] = bound[:, 0] - scale*x0
        if sum(up_bool) > len(up_bool) / thresh_ratio:
            bound[:, 1] = bound[:, 1] + scale*x0


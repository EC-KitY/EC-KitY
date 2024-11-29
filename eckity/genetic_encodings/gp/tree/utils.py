"""
This module implements some utility functions.
"""

from typing import Callable, Dict, List, Union, get_type_hints

import numpy as np

from eckity.base.utils import arity


def create_terminal_set(
    X: np.ndarray, typed=False
) -> Union[List[str], Dict[str, type]]:
    """
    Create a terminal set from a 2D-shaped numpy array.

    Example: \n
        X = array([[  4,   7,  -7, -10],  \n
                   [  7,  -3,   3,  -8],  \n
                   [  8,  -5,  -3,  -1]])

    Returns the list: \n
        ['x0', 'x1', 'x2', 'x3']

    Parameters
    ----------
    X : 2d numpy array
        The array from which we wish to extract features -- which will become terminals.

    Returns
    -------
    Terminal set (a list).

    """

    features = [f"x{i}" for i in range(X.shape[1])]
    if not typed:
        return features
    # convert numpy dtypes to python types
    return {x: type(X[0, i].item()) for i, x in enumerate(features)}


def generate_args(X: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate keyword arguments from a 2d array for passing to GPTree.execute.

    Example: \n
        X = array([[  4,   7,  -7, -10],  \n
                   [  7,  -3,   3,  -8],  \n
                   [  8,  -5,  -3,  -1]])

    Returns the dict: \n
        {'x0': array([4, 7, 8]), \n
         'x1': array([ 7, -3, -5]), \n
         'x2': array([-7,  3, -3]), \n
         'x3': array([-10,  -8,  -1])}

    Returns
    -------
    kwargs : dict
        A keyword dictionary that includes a value for every variable x_i in the terminal set.

    """

    return {f"x{i}": X[:, i] for i in range(X.shape[1])}


def get_func_types(f: Callable) -> List[type]:
    """
    Return list of function types in the following format:
    [type_arg_1, type_arg_2, ..., type_arg_n, return_type]

    Parameters
    ----------
    f : Callable
        function (builtin or user-defined)

    Returns
    -------
    List[type]
        List of function types, sorted by argument order
        with the return type as the last element.
        For untyped functions, None is used.

    Examples
    --------
    >>> def f(x: int, y: float) -> float:
    ...     return x + y
    >>> get_func_types(f)
    [int, float, float]

    >>> def f(x, y):
    ...     return x + y
    >>> get_func_types(f)
    [None, None, None]
    """
    params_types: Dict = get_type_hints(f)
    type_list = list(params_types.values())
    if not type_list:
        # If we don't have type hints, assign None
        type_list = [None] * (arity(f) + 1)
    return type_list


def get_return_type(func: Callable) -> type:
    return get_type_hints(func).get("return", None)

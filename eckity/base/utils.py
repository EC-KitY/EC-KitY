"""
This module implements some utility functions. 
"""

from inspect import getfullargspec
from typing import Callable


def arity(func: Callable) -> int:
    """
    Parameters
    ----------
    func : function
            A function.

    Returns
    -------
    arity : int
            The function's arity.
    """
    return len(getfullargspec(func)[0])

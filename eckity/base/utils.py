"""
This module implements some utility functions. 
"""

from inspect import getfullargspec


def arity(func): 
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

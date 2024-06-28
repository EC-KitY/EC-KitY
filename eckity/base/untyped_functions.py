"""
This module implements functions used in the function (internal) nodes of a GP tree.
Note: all functions work on numpy arrays.
"""

import numpy as np


def untyped_add(x, y):
    """x+y"""
    return np.add(x, y)


def untyped_sub(x, y):
    """x-y"""
    return np.subtract(x, y)


def untyped_mul(x, y):
    """x*y"""
    return np.multiply(x, y)


def untyped_div(x, y):
    """protected division: if abs(y) > 0.001 return x/y else return 0"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) > 0.001, np.divide(x, y), 0.0)


def untyped_sqrt(x):
    """protected square root: sqrt(abs(x))"""
    return np.sqrt(np.absolute(x))


def untyped_log(x):
    """protected log: if abs(x) > 0.001 return log(abs(x)) else return 0"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.0)


def untyped_abs(x):
    """absolute value of x"""
    return np.absolute(x)


def untyped_neg(x):
    """negative of x"""
    return np.negative(x)


def untyped_inv(x):
    """protected inverse: if abs(x) > 0.001 return 1/x else return 0"""
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     return np.where(np.abs(x) > 0.001, 1. / x, 0.)
    return untyped_div(1, x)


def untyped_max(x, y):
    """maximum(x,y)"""
    return np.maximum(x, y)


def untyped_min(x, y):
    """minimum(x,y)"""
    return np.minimum(x, y)


def untyped_sin(x):
    """sin(x)"""
    return np.sin(x)


def untyped_cos(x):
    """cos(x)"""
    return np.cos(x)


def untyped_tan(x):
    """tan(x)"""
    return np.tan(x)


def untyped_iflte0(x, y, z):
    """if x <= 0 return y else return z"""
    return np.where(x <= 0, y, z)


def untyped_ifgt0(x, y, z):
    """if x > 0 return y else return z"""
    return np.where(x > 0, y, z)


def untyped_iflte(x, y, z, w):
    """if x <= y return z else return w"""
    return np.where(x <= y, z, w)


def untyped_ifgt(x, y, z, w):
    """if x > y return z else return w"""
    return np.where(x > y, z, w)


def untyped_and(x, y):
    """x and y"""
    return np.bitwise_and(x, y)


def untyped_or(x, y):
    """x or y"""
    return np.bitwise_or(x, y)


def untyped_not(x):
    """not x"""
    return np.logical_not(x).astype(int)


def untyped_if_then_else(test, dit, dif):
    """if test return dit else return dif"""
    return np.where(test, dit, dif)


full_function_set = [
    untyped_add,
    untyped_sub,
    untyped_mul,
    untyped_div,
    untyped_sqrt,
    untyped_log,
    untyped_abs,
    untyped_neg,
    untyped_inv,
    untyped_max,
    untyped_min,
    untyped_sin,
    untyped_cos,
    untyped_tan,
    untyped_iflte0,
    untyped_ifgt0,
    untyped_iflte,
    untyped_ifgt,
    untyped_and,
    untyped_or,
    untyped_not,
    untyped_if_then_else,
]

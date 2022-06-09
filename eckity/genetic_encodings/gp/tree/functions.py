"""
This module implements functions used in the function (internal) nodes of a GP tree.
Note: all functions work on numpy arrays.
"""

import numpy as np


def f_add(x, y):
    """x+y"""
    return np.add(x, y)


def f_sub(x, y):
    """x-y"""
    return np.subtract(x, y)


def f_mul(x, y):
    """x*y"""
    return np.multiply(x, y)


def f_div(x, y):
    """protected division: if abs(y) > 0.001 return x/y else return 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(y) > 0.001, np.divide(x, y), 0.)


def f_sqrt(x):
    """protected square root: sqrt(abs(x))"""
    return np.sqrt(np.absolute(x))


def f_log(x):
    """protected log: if abs(x) > 0.001 return log(abs(x)) else return 0"""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.)


def f_abs(x):
    """absolute value of x"""
    return np.absolute(x)


def f_neg(x):
    """negative of x"""
    return np.negative(x)


def f_inv(x):
    """protected inverse: if abs(x) > 0.001 return 1/x else return 0"""
    # with np.errstate(divide='ignore', invalid='ignore'):
    #     return np.where(np.abs(x) > 0.001, 1. / x, 0.)
    return f_div(1, x)


def f_max(x, y):
    """maximum(x,y)"""
    return np.maximum(x, y)


def f_min(x, y):
    """minimum(x,y)"""
    return np.minimum(x, y)


def f_sin(x):
    """sin(x)"""
    return np.sin(x)


def f_cos(x):
    """cos(x)"""
    return np.cos(x)


def f_tan(x):
    """tan(x)"""
    return np.tan(x)


def f_iflte0(x, y, z):
    """if x <= 0 return y else return z"""
    return np.where(x <= 0, y, z)


def f_ifgt0(x, y, z):
    """if x > 0 return y else return z"""
    return np.where(x > 0, y, z)


def f_iflte(x, y, z, w):
    """if x <= y return z else return w"""
    return np.where(x <= y, z, w)


def f_ifgt(x, y, z, w):
    """if x > y return z else return w"""
    return np.where(x > y, z, w)


def f_and(x, y):
    """x and y"""
    return np.bitwise_and(x, y)


def f_or(x, y):
    """x or y"""
    return np.bitwise_or(x, y)


def f_not(x):
    """not x"""
    return np.logical_not(x).astype(int)


def f_if_then_else(test, dit, dif):
    """if test return dit else return dif"""
    return np.where(test, dit, dif)



full_function_set = [f_add, f_sub, f_mul, f_div, f_sqrt, f_log, f_abs, f_neg, f_inv, f_max, f_min, f_sin, f_cos, f_tan,
                     f_iflte0, f_ifgt0, f_iflte, f_ifgt, f_add, f_or, f_not, f_if_then_else]

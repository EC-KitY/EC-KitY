"""
This module implements functions used in the function (Numberernal) nodes of a GP tree.
Note: all functions work on numpy arrays.
"""

from numbers import Number
from typing import Any

import numpy as np


def typed_add(x: Number, y: Number) -> Number:
    """x+y"""
    return np.add(x, y)


def typed_sub(x: Number, y: Number) -> Number:
    """x-y"""
    return np.subtract(x, y)


def typed_mul(x: Number, y: Number) -> Number:
    """x*y"""
    return np.multiply(x, y)


def typed_div(x: Number, y: Number) -> Number:
    """protected division: if abs(y) > 0.001 return x/y else return 0"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) > 0.001, np.divide(x, y), 0.0)


def typed_sqrt(x: Number) -> Number:
    """protected square root: sqrt(abs(x))"""
    return np.sqrt(np.absolute(x))


def typed_log(x: Number) -> Number:
    """protected log: if abs(x) > 0.001 return log(abs(x)) else return 0"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.0)


def typed_abs(x: Number) -> Number:
    """absolute value of x"""
    return np.absolute(x)


def typed_neg(x: Number) -> Number:
    """negative of x"""
    return np.negative(x)


def typed_inv(x: Number) -> Number:
    """protected inverse: if abs(x) > 0.001 return 1/x else return 0"""
    return typed_div(1, x)


def typed_max(x: Number, y: Number) -> Number:
    """maximum(x,y)"""
    return np.maximum(x, y)


def typed_min(x: Number, y: Number) -> Number:
    """minimum(x,y)"""
    return np.minimum(x, y)


def typed_sin(x: Number) -> Number:
    """sin(x)"""
    return np.sin(x)


def typed_cos(x: Number) -> Number:
    """cos(x)"""
    return np.cos(x)


def typed_tan(x: Number) -> Number:
    """tan(x)"""
    return np.tan(x)


def typed_iflte0(x: Number, y: Number, z: Number) -> Number:
    """if x <= 0 return y else return z"""
    return np.where(x <= 0, y, z)


def typed_ifgt0(x: Number, y: Number, z: Number) -> Number:
    """if x > 0 return y else return z"""
    return np.where(x > 0, y, z)


def typed_iflte(x: Number, y: Number, z: Number, w: Number) -> Number:
    """if x <= y return z else return w"""
    return np.where(x <= y, z, w)


def typed_ifgt(x: Number, y: Number, z: Number, w: Number) -> Number:
    """if x > y return z else return w"""
    return np.where(x > y, z, w)


def typed_and(x: Number, y: Number) -> Number:
    """x and y"""
    return np.bitwise_and(x, y)


def typed_or(x: Number, y: Number) -> Number:
    """x or y"""
    return np.bitwise_or(x, y)


def typed_not(x: Number) -> Number:
    """not x"""
    return np.logical_not(x).astype(int)


def typed_if_then_else(test: bool, dit: Any, dif: Any) -> Number:
    """if test return dit else return dif"""
    return np.where(test, dit, dif)


full_function_set = [
    typed_add,
    typed_sub,
    typed_mul,
    typed_div,
    typed_sqrt,
    typed_log,
    typed_abs,
    typed_neg,
    typed_inv,
    typed_max,
    typed_min,
    typed_sin,
    typed_cos,
    typed_tan,
    typed_iflte0,
    typed_ifgt0,
    typed_iflte,
    typed_ifgt,
    typed_and,
    typed_or,
    typed_not,
    typed_if_then_else,
]

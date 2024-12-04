"""
This module implements functions used in the function nodes of a GP tree.
Note: all functions work on numpy arrays.
"""

from typing import Any
import numpy as np
from .types import t_argmax


def add2floats(x: float, y: float) -> float:
    """x+y"""
    return np.add(x, y)


def sub2floats(x: float, y: float) -> float:
    """x-y"""
    return np.subtract(x, y)


def mul2floats(x: float, y: float) -> float:
    """x*y"""
    return np.multiply(x, y)


def div2floats(x: float, y: float) -> float:
    """protected division: if abs(y) > 0.001 return x/y else return 0"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(y) > 0.001, np.divide(x, y), 0.0)


def sqrt_float(x: float) -> float:
    """protected square root: sqrt(abs(x))"""
    return np.sqrt(np.absolute(x))


def log_float(x: float) -> float:
    """protected log: if abs(x) > 0.001 return log(abs(x)) else return 0"""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(x) > 0.001, np.log(np.abs(x)), 0.0)


def abs_float(x: float) -> float:
    """absolute value of x"""
    return np.absolute(x)


def neg_float(x: float) -> float:
    """negative of x"""
    return np.negative(x)


def inv_float(x: float) -> float:
    """protected inverse: if abs(x) > 0.001 return 1/x else return 0"""
    return div2floats(1, x)


def max2floats(x: float, y: float) -> float:
    """maximum(x,y)"""
    return np.maximum(x, y)


def min2floats(x: float, y: float) -> float:
    """minimum(x,y)"""
    return np.minimum(x, y)


def sin_float(x: float) -> float:
    """sin(x)"""
    return np.sin(x)


def cos_float(x: float) -> float:
    """cos(x)"""
    return np.cos(x)


def tan_float(x: float) -> float:
    """tan(x)"""
    return np.tan(x)


def iflte0_floats(x: float, y: float, z: float) -> float:
    """if x <= 0 return y else return z"""
    return np.where(x <= 0, y, z)


def ifgt0_floats(x: float, y: float, z: float) -> float:
    """if x > 0 return y else return z"""
    return np.where(x > 0, y, z)


def iflte_floats(x: float, y: float, z: float, w: float) -> float:
    """if x <= y return z else return w"""
    return np.where(x <= y, z, w)


def ifgt_floats(x: float, y: float, z: float, w: float) -> float:
    """if x > y return z else return w"""
    return np.where(x > y, z, w)


def and2floats(x: float, y: float) -> float:
    """x and y"""
    return np.logical_and(x, y)


def or2floats(x: float, y: float) -> float:
    """x or y"""
    return np.logical_or(x, y)


def not2floats(x: float) -> float:
    """not x"""
    return np.logical_not(x).astype(float)


def if_then_else(test: bool, dit: Any, dif: Any) -> float:
    """if test return dit else return dif"""
    return np.where(test, dit, dif)


def argmax2floats(x0: float, x1: float) -> t_argmax:
    return np.argmax([x0, x1], axis=0)


def and2ints(x: int, y: int) -> int:
    """x and y"""
    return np.logical_and(x, y)


def or2ints(x: int, y: int) -> int:
    """x or y"""
    return np.logical_or(x, y)


def not2ints(x: int) -> int:
    """not x"""
    return np.logical_not(x).astype(int)


def if_then_else3ints(test: int, dit: int, dif: int) -> int:
    """if test return dit else return dif"""
    return np.where(test, dit, dif)


def and2bools(x: bool, y: bool) -> bool:
    """x and y"""
    return np.bitwise_and(x, y)


def or2bools(x: bool, y: bool) -> bool:
    """x or y"""
    return np.bitwise_or(x, y)


def not2bools(x: bool) -> bool:
    """not x"""
    return np.logical_not(x)


def if_then_else3bools(test: bool, dit: bool, dif: bool) -> bool:
    """if test return dit else return dif"""
    return np.where(test, dit, dif)


__all__ = [
    "add2floats",
    "sub2floats",
    "mul2floats",
    "div2floats",
    "sqrt_float",
    "log_float",
    "abs_float",
    "neg_float",
    "inv_float",
    "max2floats",
    "min2floats",
    "sin_float",
    "cos_float",
    "tan_float",
    "iflte0_floats",
    "ifgt0_floats",
    "iflte_floats",
    "ifgt_floats",
    "and2floats",
    "or2floats",
    "not2floats",
    "if_then_else",
    "argmax2floats",
    "and2ints",
    "or2ints",
    "not2ints",
    "if_then_else3ints",
    "and2bools",
    "or2bools",
    "not2bools",
    "if_then_else3bools",
]

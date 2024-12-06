import numpy as np
import pytest

from eckity.base.typed_functions import *


def test_add2floats():
    assert add2floats(1.0, 2.0) == 3.0
    assert add2floats(-1.0, -2.0) == -3.0
    assert add2floats(0.0, 0.0) == 0.0


def test_sub2floats():
    assert sub2floats(5.0, 3.0) == 2.0
    assert sub2floats(-1.0, -1.0) == 0.0
    assert sub2floats(0.0, 1.0) == -1.0


def test_mul2floats():
    assert mul2floats(3.0, 2.0) == 6.0
    assert mul2floats(-2.0, -2.0) == 4.0
    assert mul2floats(0.0, 100.0) == 0.0


def test_div2floats():
    assert div2floats(6.0, 2.0) == 3.0
    assert div2floats(1.0, 0.0) == 0.0
    assert div2floats(0.0, 1.0) == 0.0


def test_sqrt_float():
    assert sqrt_float(4.0) == 2.0
    assert sqrt_float(-4.0) == 2.0
    assert sqrt_float(0.0) == 0.0


def test_log_float():
    assert log_float(1.0) == 0.0
    assert log_float(0.0) == 0.0
    assert log_float(-1.0) == 0.0


def test_abs_float():
    assert abs_float(-3.0) == 3.0
    assert abs_float(3.0) == 3.0
    assert abs_float(0.0) == 0.0


def test_neg_float():
    assert neg_float(5.0) == -5.0
    assert neg_float(-5.0) == 5.0
    assert neg_float(0.0) == 0.0


def test_inv_float():
    assert inv_float(2.0) == 0.5
    assert inv_float(0.0) == 0.0
    assert inv_float(-2.0) == -0.5


def test_max2floats():
    assert max2floats(3.0, 5.0) == 5.0
    assert max2floats(-1.0, -2.0) == -1.0
    assert max2floats(0.0, 0.0) == 0.0


def test_min2floats():
    assert min2floats(3.0, 5.0) == 3.0
    assert min2floats(-1.0, -2.0) == -2.0
    assert min2floats(0.0, 0.0) == 0.0


def test_sin_float():
    assert np.isclose(sin_float(np.pi / 2), 1.0)
    assert np.isclose(sin_float(0.0), 0.0)
    assert np.isclose(sin_float(-np.pi / 2), -1.0)


def test_cos_float():
    assert np.isclose(cos_float(0.0), 1.0)
    assert np.isclose(cos_float(np.pi), -1.0)
    assert np.isclose(cos_float(np.pi / 2), 0.0)


def test_tan_float():
    assert np.isclose(tan_float(0.0), 0.0)
    assert np.isclose(tan_float(np.pi / 4), 1.0)
    assert np.isclose(tan_float(-np.pi / 4), -1.0)


def test_iflte0_floats():
    assert iflte0_floats(0.0, 1.0, 2.0) == 1.0
    assert iflte0_floats(-1.0, 1.0, 2.0) == 1.0
    assert iflte0_floats(1.0, 1.0, 2.0) == 2.0


def test_ifgt0_floats():
    assert ifgt0_floats(1.0, 1.0, 2.0) == 1.0
    assert ifgt0_floats(0.0, 1.0, 2.0) == 2.0
    assert ifgt0_floats(-1.0, 1.0, 2.0) == 2.0


def test_iflte_floats():
    assert iflte_floats(1.0, 2.0, 3.0, 4.0) == 3.0
    assert iflte_floats(2.0, 2.0, 3.0, 4.0) == 3.0
    assert iflte_floats(3.0, 2.0, 3.0, 4.0) == 4.0


def test_ifgt_floats():
    assert ifgt_floats(3.0, 2.0, 3.0, 4.0) == 3.0
    assert ifgt_floats(2.0, 2.0, 3.0, 4.0) == 4.0
    assert ifgt_floats(1.0, 2.0, 3.0, 4.0) == 4.0


def test_and2floats():
    assert and2floats(1.0, 1.0) == 1.0
    assert and2floats(0.0, 1.0) == 0.0
    assert and2floats(0.0, 0.0) == 0.0


def test_or2floats():
    assert or2floats(1.0, 0.0) == 1.0
    assert or2floats(0.0, 1.0) == 1.0
    assert or2floats(0.0, 0.0) == 0.0


def test_not2floats():
    assert not2floats(0.0) == 1.0
    assert not2floats(1.0) == 0.0
    assert not2floats(-1.0) == 0.0

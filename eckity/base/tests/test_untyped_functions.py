import numpy as np

from eckity.base.untyped_functions import *


def test_f_add():
    assert np.array_equal(f_add(1, 2), 3)
    assert np.array_equal(
        f_add(np.array([1, 2]), np.array([3, 4])), np.array([4, 6])
    )


def test_f_sub():
    assert np.array_equal(f_sub(5, 3), 2)
    assert np.array_equal(
        f_sub(np.array([10, 20]), np.array([5, 5])), np.array([5, 15])
    )


def test_f_mul():
    assert np.array_equal(f_mul(3, 4), 12)
    assert np.array_equal(
        f_mul(np.array([2, 3]), np.array([4, 5])), np.array([8, 15])
    )


def test_f_div():
    assert np.array_equal(f_div(6, 3), 2)
    assert np.array_equal(
        f_div(np.array([1, 2]), np.array([0.001, 0.002])),
        np.array([0, 1000]),
    )
    assert np.array_equal(f_div(1, 0), 0)
    assert np.array_equal(f_div(1, 0.0001), 0)


def test_f_sqrt():
    assert np.array_equal(f_sqrt(4), 2)
    assert np.array_equal(f_sqrt(-4), 2)
    assert np.array_equal(f_sqrt(np.array([4, -9])), np.array([2, 3]))


def test_f_log():
    assert np.array_equal(f_log(1), 0)
    assert np.array_equal(f_log(0), 0)
    assert np.array_equal(
        f_log(np.array([0.001, 10])), np.array([0, np.log(10)])
    )


def test_f_abs():
    assert np.array_equal(f_abs(-5), 5)
    assert np.array_equal(f_abs(np.array([-1, 2, -3])), np.array([1, 2, 3]))


def test_f_neg():
    assert np.array_equal(f_neg(5), -5)
    assert np.array_equal(f_neg(np.array([1, -2, 3])), np.array([-1, 2, -3]))


def test_f_inv():
    assert np.array_equal(f_inv(2), 0.5)
    assert np.array_equal(f_inv(0), 0)
    assert np.array_equal(f_inv(np.array([2, 0.0001])), np.array([0.5, 0]))


def test_f_max():
    assert np.array_equal(f_max(1, 2), 2)
    assert np.array_equal(
        f_max(np.array([1, 3]), np.array([2, 2])), np.array([2, 3])
    )


def test_f_min():
    assert np.array_equal(f_min(1, 2), 1)
    assert np.array_equal(
        f_min(np.array([1, 3]), np.array([2, 2])), np.array([1, 2])
    )


def test_f_sin():
    assert np.allclose(f_sin(0), 0)
    assert np.allclose(f_sin(np.pi / 2), 1)


def test_f_cos():
    assert np.allclose(f_cos(0), 1)
    assert np.allclose(f_cos(np.pi), -1)


def test_f_tan():
    assert np.allclose(f_tan(0), 0)
    assert np.allclose(f_tan(np.pi / 4), 1)


def test_f_iflte0():
    assert np.array_equal(f_iflte0(-1, 2, 3), 2)
    assert np.array_equal(f_iflte0(1, 2, 3), 3)


def test_f_ifgt0():
    assert np.array_equal(f_ifgt0(1, 2, 3), 2)
    assert np.array_equal(f_ifgt0(-1, 2, 3), 3)


def test_f_iflte():
    assert np.array_equal(f_iflte(1, 2, 3, 4), 3)
    assert np.array_equal(f_iflte(3, 2, 3, 4), 4)


def test_f_ifgt():
    assert np.array_equal(f_ifgt(3, 2, 3, 4), 3)
    assert np.array_equal(f_ifgt(1, 2, 3, 4), 4)


def test_f_and():
    assert np.array_equal(f_and(1, 0), 0)
    assert np.array_equal(
        f_and(np.array([1, 0]), np.array([0, 1])), np.array([0, 0])
    )


def test_f_or():
    assert np.array_equal(f_or(1, 0), 1)
    assert np.array_equal(
        f_or(np.array([1, 0]), np.array([0, 1])), np.array([1, 1])
    )


def test_f_not():
    assert np.array_equal(f_not(1), 0)
    assert np.array_equal(f_not(np.array([1, 0])), np.array([0, 1]))


def test_f_if_then_else():
    assert np.array_equal(f_if_then_else(True, 1, 0), 1)
    assert np.array_equal(f_if_then_else(False, 1, 0), 0)
    assert np.array_equal(
        f_if_then_else(
            np.array([True, False]), np.array([1, 2]), np.array([3, 4])
        ),
        np.array([1, 4]),
    )

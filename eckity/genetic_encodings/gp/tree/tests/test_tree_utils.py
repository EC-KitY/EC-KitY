import numpy as np
import pytest
from eckity.genetic_encodings.gp.tree.utils import (
    create_terminal_set,
    generate_args,
)


@pytest.mark.parametrize(
    "X, typed, expected",
    [
        (
            np.array([[4, 7, -7, -10], [7, -3, 3, -8], [8, -5, -3, -1]]),
            False,
            ["x0", "x1", "x2", "x3"],
        ),
        (
            np.array([[4, 7, -7, -10], [7, -3, 3, -8], [8, -5, -3, -1]]),
            True,
            {"x0": int, "x1": int, "x2": int, "x3": int},
        ),
    ],
)
def test_create_terminal_set(X, typed, expected):
    assert create_terminal_set(X, typed) == expected


def test_generate_args():
    X = np.array([[4, 7, -7, -10], [7, -3, 3, -8], [8, -5, -3, -1]])
    expected = {
        "x0": np.array([4, 7, 8]),
        "x1": np.array([7, -3, -5]),
        "x2": np.array([-7, 3, -3]),
        "x3": np.array([-10, -8, -1]),
    }
    args = generate_args(X)
    assert args.keys() == expected.keys()
    for key in args:
        assert np.array_equal(args[key], expected[key])

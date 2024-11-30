import pytest

from eckity.base.utils import arity
from eckity.base.typed_functions import add2floats, or2floats
from eckity.base.untyped_functions import f_mul, f_not


@pytest.mark.parametrize(
    "func, expected",
    [
        (add2floats, 2),
        (or2floats, 2),
        (f_mul, 2),
        (f_not, 1),
    ]
)
def test_arity(func, expected):
    assert arity(func) == expected

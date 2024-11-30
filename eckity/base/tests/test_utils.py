import pytest

from eckity.base.utils import arity
from eckity.base.typed_functions import typed_add, typed_or
from eckity.base.untyped_functions import f_mul, f_not

@pytest.mark.parametrize(
    "func, expected"
    [
        (typed_add, 2),
        (typed_or, 2),
        (f_mul, 2),
        (f_not, 1),
    ]
)
def test_arity(func, expected):
    assert arity(func) == expected

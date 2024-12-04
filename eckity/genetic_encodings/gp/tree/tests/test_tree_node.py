import pytest

from eckity.base.typed_functions import add2floats
from eckity.base.untyped_functions import f_add
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode
from eckity.genetic_encodings.gp.tree.utils import get_func_types


@pytest.mark.parametrize(
    "function, expected_types",
    [
        (add2floats, [float, float, float]),
        (f_add, [None, None, None]),
    ],
)
def test_get_func_types(function, expected_types):
    """
    Test that get_func_types returns the correct function types
    """
    func_types = get_func_types(function)
    assert func_types == expected_types


def test_missing_type_hints():
    """
    define some bad functions
    not using pytest decorator because lambda expressions have no type hints
    """

    # missing return type hint
    def add_no_return_type(x: int, y: int):
        return x + y

    # missing x type hint
    def add_no_x_type(x, y: int) -> int:
        return x + y

    # missing y type hint
    def add_no_y_type(x: int, y) -> int:
        return x + y

    for func in [add_no_return_type, add_no_x_type, add_no_y_type]:
        with pytest.raises(ValueError) as excinfo:
            FunctionNode(func)
        assert "missing type hints" in str(excinfo)


@pytest.mark.parametrize(
    "node1, node2, expected",
    [
        (TerminalNode(1, int), TerminalNode(1, int), True),
        (TerminalNode(1, int), TerminalNode(True, bool), False),
        (TerminalNode(0, int), TerminalNode(False, bool), False),
        (TerminalNode(1, int), TerminalNode(2, int), False),
        (TerminalNode(1), TerminalNode(1), True),
        (TerminalNode(1), TerminalNode(True), False),
    ],
)
def test_eq(node1, node2, expected):
    assert (node1 == node2) == expected
    assert (node2 == node1) == expected

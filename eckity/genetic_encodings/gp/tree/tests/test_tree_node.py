from types import NoneType

import pytest

from eckity.base.typed_functions import add2floats
from eckity.base.untyped_functions import f_add
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode


@pytest.mark.parametrize(
    "function, expected_types",
    [
        (add2floats, [float, float, float]),
        (f_add, [NoneType, NoneType, NoneType]),
    ],
)
def test_get_func_types(function, expected_types):
    """
    Test that get_func_types returns the correct function types
    """
    func_types = FunctionNode.get_func_types(function)
    assert func_types == expected_types


@pytest.mark.parametrize(
    "node, expected",
    [
        (TerminalNode(1, int), 1),
        (TerminalNode(1), 1),
        (
            FunctionNode(
                add2floats
            ),
            2.0,
        ),
        (
            FunctionNode(
                f_add,
            ),
            2,
        ),
    ],
)
def test_execute(node, expected):
    assert node.execute(x=1.0, y=1.0) == expected


def test_execute_bad_type():
    """
    Test that execute raises a TypeError if the input type is incorrect
    """
    node = TerminalNode("x", int)
    with pytest.raises(TypeError) as excinfo:
        node.execute(x=1.0)
    assert f"type {node.node_type}" in str(excinfo)


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
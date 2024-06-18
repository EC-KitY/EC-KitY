import pytest
from functions import typed_add, untyped_add

from eckity.genetic_encodings.gp import FunctionNode, TerminalNode


@pytest.mark.parametrize(
    "function, expected_types",
    [
        (typed_add, [int, int, int]),
        (untyped_add, []),
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
                typed_add,
                children=[TerminalNode(1, int), TerminalNode(1, int)],
            ),
            2,
        ),
        (
            FunctionNode(
                untyped_add,
                children=[TerminalNode(1), TerminalNode(1)],
            ),
            2,
        ),
    ],
)
def test_depth(node, expected):
    assert node.depth() == expected


@pytest.mark.parametrize(
    "node, expected",
    [
        (TerminalNode(1, int), 1),
        (TerminalNode(1), 1),
        (
            FunctionNode(
                typed_add,
                children=[TerminalNode("x", int), TerminalNode(1, int)],
            ),
            2,
        ),
        (
            FunctionNode(
                untyped_add,
                children=[TerminalNode("x"), TerminalNode(1)],
            ),
            2,
        ),
    ],
)
def test_execute(node, expected):
    assert node.execute(x=1) == expected

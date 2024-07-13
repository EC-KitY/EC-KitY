from typing import Any, Callable, Dict, List, Union

import pytest

from eckity.base.typed_functions import sqrt_float
from eckity.base.untyped_functions import f_log
from eckity.creators import GrowCreator
from eckity.genetic_encodings.gp import (
    FunctionNode,
    TerminalNode,
    Tree,
    TreeNode,
)
from types import NoneType


@pytest.mark.parametrize(
    "function_set, terminal_set, expected",
    [
        (
            [sqrt_float],
            {1.0: float},
            FunctionNode(sqrt_float, children=[TerminalNode(1.0, float)]),
        ),
        (
            [f_log],
            [1.0],
            FunctionNode(f_log, children=[TerminalNode(1.0)]),
        ),
    ],
)
def test_add_children(
    function_set: List[Callable],
    terminal_set: Union[Dict[Any, type], List[Any]],
    expected: TreeNode,
):
    grow_creator = GrowCreator(init_depth=(1, 1))
    tree_ind = Tree(
        function_set=function_set,
        terminal_set=terminal_set,
    )
    node_type = float if isinstance(terminal_set, dict) else NoneType
    node = tree_ind.random_function_node(node_type=node_type)
    depth = 0
    grow_creator._add_children(node, tree_ind, depth)
    assert node == expected

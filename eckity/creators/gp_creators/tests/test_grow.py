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


@pytest.mark.parametrize(
    "function_set, terminal_set, expected",
    [
        (
            [sqrt_float],
            {1.0: float},
            [FunctionNode(sqrt_float), TerminalNode(1.0, float)],
        ),
        (
            [f_log],
            [1.0],
            [FunctionNode(f_log), TerminalNode(1.0)],
        ),
    ],
)
def test_add_children(
    function_set: List[Callable],
    terminal_set: Union[Dict[Any, type], List[Any]],
    expected: List[TreeNode],
):
    grow_creator = GrowCreator(
        init_depth=(1, 1),
        function_set=function_set,
        terminal_set=terminal_set,
    )
    
    node_type = float if isinstance(terminal_set, dict) else None
    tree_ind = Tree(
        function_set=function_set,
        terminal_set=terminal_set,
        erc_range=None,
        root_type=node_type,
    )

    node = tree_ind.random_function(node_type=node_type)
    tree_ind.add_tree(node)

    grow_creator._add_children(
        tree_ind.tree,
        node,
        tree_ind.random_function,
        tree_ind.random_terminal,
        depth=0,
    )
    assert tree_ind.tree == expected

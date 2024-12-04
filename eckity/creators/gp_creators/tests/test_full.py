from ....base.untyped_functions import f_add
from ....fitness import SimpleFitness
from ....genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from ..full import FullCreator


def test_create_tree():
    init_depth = (1, 1)
    function_set = [f_add]
    terminal_set = ["x"]

    creator = FullCreator(init_depth, function_set, terminal_set)
    tree_ind = Tree(
        SimpleFitness(),
        function_set=function_set,
        terminal_set=terminal_set,
        erc_range=None,
    )
    creator.create_tree(
        tree_ind.tree, tree_ind.random_function, tree_ind.random_terminal
    )

    assert tree_ind.tree == [
        FunctionNode(function_set[0]),
        TerminalNode(terminal_set[0]),
        TerminalNode(terminal_set[0]),
    ]

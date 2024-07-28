from ....fitness import SimpleFitness
from ....genetic_encodings.gp import Tree, TerminalNode, FunctionNode
from ..full import FullCreator
from ....base.untyped_functions import f_add


def test_create_tree():
    init_depth = (1, 1)
    function_set = [f_add]
    terminal_set = ["x"]

    creator = FullCreator(init_depth)
    tree_ind = Tree(
        SimpleFitness(),
        function_set=function_set,
        terminal_set=terminal_set,
        erc_range=None
    )
    creator.create_tree(tree_ind)

    assert tree_ind.tree == [
        FunctionNode(function_set[0]),
        TerminalNode(terminal_set[0]),
        TerminalNode(terminal_set[0]),
    ]

import pytest

from eckity.fitness import GPFitness
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from eckity.genetic_operators import SubtreeMutation


def equal2ints(a: int, b: int) -> bool:
    return a == b


def inc_int(x: int) -> int:
    return x + 1


def f_equal(a, b):
    return a == b


@pytest.mark.parametrize(
    "tree",
    [
        Tree(
            fitness=GPFitness(),
            function_set=[equal2ints, inc_int],
            terminal_set={"x": int, "y": int},
            tree=[
                FunctionNode(function=equal2ints),
                TerminalNode(value=1, node_type=int),
                TerminalNode(value=2, node_type=int),
            ],
            root_type=bool,
        ),
        Tree(
            fitness=GPFitness(),
            function_set=[f_equal],
            terminal_set=["x", "y"],
            tree=[
                FunctionNode(function=equal2ints),
                TerminalNode(value=1),
                TerminalNode(value=2),
            ],
        ),
    ],
)
def test_subtree_mutation_success(tree):
    subtree_mutation = SubtreeMutation(probability=1.0)

    tree_copy = tree.clone()

    # Perform subtree crossover
    subtree_mutation.apply([tree_copy])

    assert tree.root == tree_copy.root
    assert tree != tree_copy

import pytest
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from eckity.genetic_operators import SubtreeCrossover
from eckity.fitness import GPFitness


# Define custom functions
def typed_add_inc(n: int, inc: bool) -> int:
    return n + 1 if inc else n


def untyped_add_inc(n, inc):
    return n + 1 if inc else n


@pytest.mark.parametrize(
    "tree1, tree2",
    [
        (
            Tree(
                fitness=GPFitness(),
                function_set=[typed_add_inc],
                terminal_set={8: int, True: bool, False: bool},
                root=FunctionNode(
                    typed_add_inc,
                    children=[TerminalNode(8, int), TerminalNode(True, bool)],
                ),
            ),
            Tree(
                fitness=GPFitness(),
                function_set=[typed_add_inc],
                terminal_set={8: int, True: bool, False: bool},
                root=FunctionNode(
                    typed_add_inc,
                    children=[TerminalNode(8, int), TerminalNode(False, bool)],
                ),
            ),
        ),
    ],
)
def test_subtree_crossover_typed(tree1, tree2):
    subtree_crossover = SubtreeCrossover(node_type=bool, probability=1.0)

    for _ in range(10):
        tree1_copy = tree1.clone()
        tree2_copy = tree2.clone()

        # Perform subtree crossover
        subtree_crossover.apply([tree1_copy, tree2_copy])

        # Check that the boolean nodes were swapped
        assert tree1_copy.root.children[1].value is False
        assert tree2_copy.root.children[1].value is True


@pytest.mark.parametrize(
    "tree1, tree2",
    [
        (
            Tree(
                fitness=GPFitness(),
                function_set=[untyped_add_inc],
                terminal_set=[1, True, False],
                root=FunctionNode(
                    untyped_add_inc,
                    children=[TerminalNode(1), TerminalNode(True)],
                ),
            ),
            Tree(
                fitness=GPFitness(),
                function_set=[untyped_add_inc],
                terminal_set=[1, True, False],
                root=FunctionNode(
                    untyped_add_inc,
                    children=[TerminalNode(2), TerminalNode(False)],
                ),
            ),
        ),
    ],
)
def test_subtree_crossover_untyped(tree1, tree2):
    subtree_crossover = SubtreeCrossover(probability=1.0)

    for i in range(10):
        tree1_copy = tree1.clone()
        tree2_copy = tree2.clone()

        # Perform subtree crossover
        subtree_crossover.apply([tree1_copy, tree2_copy])

        # Check that the boolean nodes were swapped
        assert i == i and tree1_copy.root != tree1.root
        assert i == i and tree2_copy.root != tree2.root
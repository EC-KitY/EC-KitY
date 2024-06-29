import pytest
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from eckity.genetic_operators import SubtreeMutation
from eckity.fitness import GPFitness
from eckity.base.typed_functions import typed_add


# define custom types
class DummyInt(int):
    pass


def dummy_add(a: int, b: int) -> DummyInt:
    return DummyInt(a + b)


@pytest.mark.parametrize(
    "tree",
    [
        Tree(
            fitness=GPFitness(),
            function_set=[dummy_add],
            terminal_set=[1, 2],
            root=FunctionNode(function=dummy_add),
        ),
        Tree(
            fitness=GPFitness(),
            function_set=[dummy_add],
            terminal_set={2: DummyInt},
            root=TerminalNode(value=1, node_type=DummyInt),
        ),
    ],
)
def test_typed_root_changed(tree):
    subtree_mutation = SubtreeMutation(node_type=DummyInt, probability=1.0)

    for _ in range(10):
        tree_copy = tree.clone()

        # Perform subtree crossover
        subtree_mutation.apply([tree_copy])

        # Check that the root function was changed
        assert tree_copy.root is not tree.root

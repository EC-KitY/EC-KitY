import pytest
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from eckity.genetic_operators import SubtreeMutation
from eckity.fitness import GPFitness


# define custom types
class DummyInt(int):
    pass


def typed_dummy_add(a: int, b: int) -> DummyInt:
    return DummyInt(a + b)


def typed_dummy_floor(a: float) -> DummyInt:
    return DummyInt(a // 1)


def untyped_dummy_add(a, b):
    return DummyInt(a + b)


def untyped_dummy_floor(a):
    return DummyInt(a // 1)


@pytest.mark.parametrize(
    "tree",
    [
        Tree(
            fitness=GPFitness(),
            function_set=[typed_dummy_add],
            terminal_set={i: int for i in range(1, 11)},
            root=FunctionNode(
                function=typed_dummy_add,
                children=[
                    TerminalNode(value=1, node_type=int),
                    TerminalNode(value=2, node_type=int),
                ],
            ),
        ),
        Tree(
            fitness=GPFitness(),
            function_set=[typed_dummy_add],
            terminal_set={2: DummyInt},
            root=TerminalNode(value=1, node_type=DummyInt),
        ),
    ],
)
def test_typed_root_changed(tree):
    subtree_mutation = SubtreeMutation(node_type=DummyInt, probability=1.0)

    for _ in range(5):
        tree_copy = tree.clone()

        # Perform subtree crossover
        subtree_mutation.apply([tree_copy])

        # Check that the root function was changed
        assert tree_copy.root is not tree.root


@pytest.mark.parametrize(
    "tree",
    [
        Tree(
            fitness=GPFitness(),
            function_set=[untyped_dummy_add],
            terminal_set=list(range(1, 11)),
            root=FunctionNode(
                function=untyped_dummy_add,
                children=[
                    TerminalNode(value=1),
                    TerminalNode(value=2),
                ],
            ),
        ),
        Tree(
            fitness=GPFitness(),
            function_set=[untyped_dummy_add],
            terminal_set=[2],
            root=TerminalNode(value=1),
        ),
    ],
)
def test_untyped_root_changed(tree):
    subtree_mutation = SubtreeMutation(probability=1.0)

    for _ in range(5):
        tree_copy = tree.clone()

        # Perform subtree crossover
        subtree_mutation.apply([tree_copy])

        # Check that the root function was changed
        assert tree_copy.root is not tree.root


@pytest.mark.parametrize(
    "tree",
    [
        Tree(
            fitness=GPFitness(),
            function_set=[typed_dummy_floor],
            terminal_set={1.0: float, 2.0: float},
            root=FunctionNode(
                function=typed_dummy_floor,
                children=[TerminalNode(value=1.0, node_type=float)],
            ),
        ),
        Tree(
            fitness=GPFitness(),
            function_set=[typed_dummy_floor],
            terminal_set={2: DummyInt},
            root=TerminalNode(value=1.0, node_type=float),
        ),
    ],
)
def test_typed_root_unchanged(tree):
    subtree_mutation = SubtreeMutation(node_type=int, probability=1.0)

    tree_copy = tree.clone()

    # Perform subtree crossover
    subtree_mutation.apply([tree_copy])

    # Check that the root function was changed
    assert tree_copy.root == tree.root

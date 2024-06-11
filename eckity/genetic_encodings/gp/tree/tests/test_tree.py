import pytest

from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.gp import Tree, TerminalNode, FunctionNode
from functions import (
    typed_add,
    typed_sub,
    typed_mul,
    typed_div,
    untyped_add,
    untyped_sub,
    untyped_mul,
    untyped_div,
)


class TestTree:
    """
    This class contains test cases for Tree and TreeNode classes
    """

    typed_functions = [typed_add, typed_sub, typed_mul, typed_div]
    untyped_functions = [untyped_add, untyped_sub, untyped_mul, untyped_div]
    terminals = ["x", "y", 0, 1, -1.0, 2.0]
    typed_tree = Tree(
        fitness=GPFitness(),
        function_set=typed_functions,
        terminal_set=terminals,
    )
    untyped_tree = Tree(
        fitness=GPFitness(),
        function_set=untyped_functions,
        terminal_set=terminals,
    )

    @pytest.fixture
    def teardown(self) -> None:
        """
        Empties both trees between tests
        Returns
        -------
        None
        """
        self.typed_tree.empty_tree()
        self.untyped_tree.empty_tree()

    def test_add_child_typed(self):
        """
        Test that add_child method adds child to the tree
        """
        typed_child = FunctionNode(typed_add)
        self.typed_tree.add_child(typed_child)
        assert self.typed_tree.root == typed_child
        assert self.typed_tree.root.node_type == int

    def test_add_child_untyped(self):
        """
        Test that add_child method adds child to the tree
        """
        untyped_child = FunctionNode(untyped_add)
        self.untyped_tree.add_child(untyped_child)
        assert self.untyped_tree.root == untyped_child
        assert self.untyped_tree.root.node_type is None

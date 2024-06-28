from numbers import Number

import pytest

from eckity.base.typed_functions import (
    typed_add,
    typed_div,
    typed_mul,
    typed_sub,
)
from eckity.base.untyped_functions import (
    untyped_add,
    untyped_div,
    untyped_mul,
    untyped_sub,
)
from eckity.base.utils import arity
from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from types import NoneType


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
    def setup(self) -> None:
        """
        Empties both trees before each test
        Returns
        -------
        None
        """
        self.typed_tree.empty_tree()
        self.untyped_tree.empty_tree()

    def test_add_child_typed(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        typed_child = FunctionNode(typed_add)
        self.typed_tree.add_child(typed_child)
        assert self.typed_tree.root == typed_child
        assert self.typed_tree.root.node_type == Number

    def test_add_child_untyped(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        untyped_child = FunctionNode(untyped_add)
        self.untyped_tree.add_child(untyped_child)
        assert self.untyped_tree.root == untyped_child
        assert self.untyped_tree.root.node_type is NoneType

    @pytest.mark.parametrize(
        "typed, root, child",
        [
            (
                True,
                FunctionNode(typed_add),
                TerminalNode(1, int),
            ),
            (
                False,
                FunctionNode(untyped_add),
                TerminalNode(1),
            ),
        ],
    )
    def test_add_child_too_many_children(self, setup, typed, root, child):
        """
        Test that add_child raises ValueError when too many children are added
        """
        tree = self.typed_tree if typed else self.untyped_tree
        tree.add_child(root)

        # add all children
        for _ in range(arity(root.function)):
            tree.add_child(child, root)

        # add one more child
        with pytest.raises(ValueError):
            tree.add_child(child, root)

    @pytest.mark.parametrize(
        "typed, root, child",
        [
            (
                True,
                FunctionNode(typed_add),
                TerminalNode("1", str),
            ),
            (
                True,
                FunctionNode(typed_add),
                TerminalNode(1, NoneType),
            ),
            (
                False,
                FunctionNode(untyped_add),
                TerminalNode(1, int),
            ),
        ],
    )
    def test_add_child_bad_type(self, setup, typed, root, child):
        tree = self.typed_tree if typed else self.untyped_tree
        tree.add_child(root)

        with pytest.raises(TypeError) as e:
            tree.add_child(child, root)
        assert "subtype" in str(e.value)

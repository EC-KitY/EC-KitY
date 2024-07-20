from numbers import Number

import pytest

from eckity.base.typed_functions import (
    add2floats,
    div2floats,
    mul2floats,
    sub2floats,
)
from eckity.base.untyped_functions import (
    f_add,
    f_div,
    f_mul,
    f_sub,
)
from eckity.base.utils import arity
from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree
from types import NoneType


class TestTree:
    """
    This class contains test cases for Tree and TreeNode classes
    """

    typed_functions = [add2floats, sub2floats, mul2floats, div2floats]
    typed_terminals = {
        "x": float,
        "y": float,
        0: int,
        1: int,
        -1.0: float,
        2.0: float,
    }

    untyped_functions = [f_add, f_sub, f_mul, f_div]
    untyped_terminals = ["x", "y", 0, 1, -1.0, 2.0]
    typed_tree = Tree(
        fitness=GPFitness(),
        function_set=typed_functions,
        terminal_set=typed_terminals,
    )
    untyped_tree = Tree(
        fitness=GPFitness(),
        function_set=untyped_functions,
        terminal_set=untyped_terminals,
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

    def test_add_child_root_typed(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        typed_child = TerminalNode(1.0, float)
        self.typed_tree.add_child(typed_child)
        assert self.typed_tree.size == 1
        assert self.typed_tree.root == typed_child
        assert self.typed_tree.root.node_type is float

    def test_add_child_root_untyped(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        untyped_child = TerminalNode(1)
        self.untyped_tree.add_child(untyped_child)
        assert self.untyped_tree.root == untyped_child
        assert self.untyped_tree.size == 1
        assert self.untyped_tree.root.node_type is NoneType

    def test_add_child_inner_typed(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        typed_parent = FunctionNode(add2floats)
        children = [TerminalNode(float(i), float) for i in range(2)]

        self.typed_tree.add_child(typed_parent)
        assert self.typed_tree.size == 1
        assert self.typed_tree.root == typed_parent
        assert self.typed_tree.root.node_type is float

        for i, child in enumerate(children):
            self.typed_tree.add_child(child, typed_parent)
            assert self.typed_tree.size == i + 2
            assert self.typed_tree.root.children[i] == child
            assert self.typed_tree.root.children[i].node_type is float

    def test_add_child_inner_untyped(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        untyped_child = TerminalNode(1)
        self.untyped_tree.add_child(untyped_child)
        assert self.untyped_tree.root == untyped_child
        assert self.untyped_tree.size == 1
        assert self.untyped_tree.root.node_type is NoneType

    @pytest.mark.parametrize(
        "typed, root, child",
        [
            (
                True,
                FunctionNode(add2floats),
                TerminalNode(1.0, float),
            ),
            (
                False,
                FunctionNode(f_add),
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
                FunctionNode(add2floats),
                TerminalNode("1", str),
            ),
            (
                True,
                FunctionNode(add2floats),
                TerminalNode(1, NoneType),
            ),
            (
                False,
                FunctionNode(f_add),
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

    def test_replace_subtree(self, setup):
        """
        Test that replace_subtree replaces subtree with another subtree
        """

        self.typed_tree.tree = (
            [
                FunctionNode(add2floats),
                TerminalNode(1.0, float),
                TerminalNode(2.0, float),
            ],
        )

        new_subtree = (
            [
                FunctionNode(sub2floats),
                TerminalNode(3.0, float),
                TerminalNode(4.0, float),
            ],
        )

        old_subtree = self.typed_tree.root.children[0]

        self.typed_tree.replace_subtree(old_subtree, new_subtree)
        assert self.typed_tree.root.children[0] == new_subtree
        assert self.typed_tree.size == 5

    @pytest.mark.parametrize(
        "typed, node, expected",
        [
            (True, [TerminalNode(1, int)], 0),
            (False, TerminalNode(1), 0),
            (
                True,
                [
                    FunctionNode(add2floats),
                    TerminalNode(1.0, float),
                    TerminalNode(2.0, float),
                ],
                1,
            ),
            (
                False,
                [
                    FunctionNode(f_add),
                    TerminalNode(1),
                    TerminalNode(2),
                ],
                1,
            ),
            (
                True,
                [
                    FunctionNode(add2floats),
                    FunctionNode(add2floats),
                    TerminalNode(1.0, float),
                    TerminalNode(2.0, float),
                    TerminalNode(2.0, float),
                ],
                2,
            ),
            (
                False,
                [
                    FunctionNode(f_add),
                    FunctionNode(f_add),
                    TerminalNode(1),
                    TerminalNode(2),
                    TerminalNode(2),
                ],
                2,
            ),
        ],
    )
    def test_depth_typed(self, setup, typed, node, expected):
        tree = self.typed_tree if typed else self.untyped_tree
        tree.tree = node
        assert tree.depth() == expected

    @pytest.mark.parametrize(
        "typed, node, expected",
        [
            (
                True,
                [TerminalNode(1, int)],
                """
                def func_0(x: float, y: float) -> int:
                    return 1
                """,
            ),
            (
                False,
                TerminalNode(1),
                """
                def func_1(x, y, z):
                    return f_add(1.0, 2.0)
                """,
            ),
            (
                True,
                [
                    FunctionNode(add2floats),
                    TerminalNode(1.0, float),
                    TerminalNode(2.0, float),
                ],
                """
                def func_0(x: float, y: float, z: float) -> float:
                    return f_add(1.0, 2.0)
                """,
            ),
            (
                False,
                [
                    FunctionNode(f_add),
                    TerminalNode(1),
                    TerminalNode(2),
                ],
                """
                def func_1(x, y):
                    return f_add(1, 2)
                """,
            ),
        ],
    )
    def test_str(self, setup, typed, node, expected):
        """
        Test that str method returns a string representation of the tree
        """
        tree = self.typed_tree if typed else self.untyped_tree
        tree.tree = (
            [
                FunctionNode(add2floats),
                TerminalNode(1.0, float),
                TerminalNode(2.0, float),
            ],
        )

        assert str(self.typed_tree) == expected

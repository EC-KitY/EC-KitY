from typing import List, Optional

import pytest

from eckity.base.typed_functions import (
    add2floats,
    div2floats,
    mul2floats,
    sub2floats,
    or2bools,
    and2bools,
)
from eckity.base.untyped_functions import f_add, f_div, f_mul, f_sub
from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.gp import (
    FunctionNode,
    TerminalNode,
    Tree,
    TreeNode,
)


class TestTree:
    """
    This class contains test cases for Tree and TreeNode classes
    """

    typed_functions = [add2floats, sub2floats, mul2floats, div2floats]
    typed_terminals = {"x": float, "y": float}

    untyped_functions = [f_add, f_sub, f_mul, f_div]
    untyped_terminals = ["x", "y"]
    typed_tree = Tree(
        fitness=GPFitness(),
        function_set=typed_functions,
        terminal_set=typed_terminals,
        root_type=float,
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

    def test_add_tree_root_typed(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        typed_child = TerminalNode(1.0, float)
        self.typed_tree.add_tree(typed_child)
        assert self.typed_tree.size() == 1
        assert self.typed_tree.root == typed_child
        assert self.typed_tree.root.node_type is float

    def test_add_tree_root_untyped(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        untyped_child = TerminalNode(1)
        self.untyped_tree.add_tree(untyped_child)
        assert self.untyped_tree.root == untyped_child
        assert self.untyped_tree.size() == 1
        assert self.untyped_tree.root.node_type is None

    def test_add_tree_inner_typed(self, setup):
        """
        Test that add_child method adds child to the tree
        """
        typed_parent = FunctionNode(add2floats)
        children = [TerminalNode(float(i), float) for i in range(2)]

        self.typed_tree.add_tree(typed_parent)
        assert self.typed_tree.size() == 1
        assert self.typed_tree.root == typed_parent
        assert self.typed_tree.root.node_type is float

        for i, child in enumerate(children):
            self.typed_tree.add_tree(child)
            assert self.typed_tree.size() == i + 2
            assert self.typed_tree.tree[i + 1] == child
            assert self.typed_tree.tree[i + 1].node_type is float

    def test_add_tree_inner_untyped(self, setup):
        """
        Test that add_tree method adds child to the tree
        """
        untyped_child = TerminalNode(1)
        self.untyped_tree.add_tree(untyped_child)
        assert self.untyped_tree.root == untyped_child
        assert self.untyped_tree.size() == 1
        assert self.untyped_tree.root.node_type is None

    @pytest.mark.parametrize(
        "root, child",
        [
            (
                FunctionNode(add2floats),
                TerminalNode("1", str),
            ),
            (
                FunctionNode(add2floats),
                TerminalNode(1, None),
            ),
            (
                FunctionNode(f_add),
                TerminalNode(1, int),
            ),
        ],
    )
    def test_add_tree_bad_type(self, root, child):
        tree = Tree(
            erc_range=None,
            function_set=[add2floats],
            terminal_set={"x": float},
            root_type=float,
        )
        tree.add_tree(root)

        with pytest.raises(ValueError) as e:
            tree.add_tree(child)
        assert "Could not add node" in str(e.value)

    def test_replace_subtree(self, setup):
        """
        Test that replace_subtree replaces subtree with another subtree
        """

        self.typed_tree.tree = list(
            [
                FunctionNode(add2floats),
                TerminalNode(1.0, float),
                TerminalNode(2.0, float),
            ],
        )

        new_subtree = list(
            [
                FunctionNode(sub2floats),
                TerminalNode(3.0, float),
                TerminalNode(4.0, float),
            ],
        )

        old_subtree = [self.typed_tree.tree[1]]

        self.typed_tree.replace_subtree(old_subtree, new_subtree)
        assert self.typed_tree.tree[1 : 1 + len(new_subtree)] == new_subtree
        assert self.typed_tree.size() == 5

    @pytest.mark.parametrize(
        "typed, tree, expected",
        [
            (True, [TerminalNode(1, int)], 0),
            (False, [TerminalNode(1)], 0),
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
    def test_depth(
        self, setup, typed: bool, tree: List[TerminalNode], expected: bool
    ):
        tree_ind = self.typed_tree if typed else self.untyped_tree
        tree_ind.tree = tree
        assert tree_ind.depth() == expected

    @pytest.mark.parametrize(
        "typed, tree, expected",
        [
            (
                True,
                [TerminalNode(1, int)],
                f"def func_{typed_tree.id}(x: float, y: float) -> int:\n  return 1",
            ),
            (
                False,
                [TerminalNode(1)],
                f"def func_{untyped_tree.id}(x, y):\n  return 1",
            ),
            (
                True,
                [
                    FunctionNode(add2floats),
                    TerminalNode(1.0, float),
                    TerminalNode(2.0, float),
                ],
                f"def func_{typed_tree.id}(x: float, y: float) -> float:\n  return add2floats(\n    1.0,\n    2.0\n  )"
            ),
            (
                False,
                [
                    FunctionNode(f_add),
                    TerminalNode(1),
                    TerminalNode(2),
                ],
                f"def func_{untyped_tree.id}(x, y):\n  return f_add(\n    1,\n    2\n  )"
            ),
            (
                True,
                [
                    FunctionNode(add2floats),
                    TerminalNode(1.0, float),
                    FunctionNode(sub2floats),
                    TerminalNode(2.0, float),
                    TerminalNode(3.0, float),
                ],
                f"def func_{typed_tree.id}(x: float, y: float) -> float:\n  return add2floats(\n    1.0,\n    sub2floats(\n      2.0,\n      3.0\n    )\n  )"
            ),
            (
                False,
                [
                    FunctionNode(f_add),
                    TerminalNode(1),
                    FunctionNode(f_sub),
                    TerminalNode(2),
                    TerminalNode(3),
                ],
                f"def func_{untyped_tree.id}(x, y):\n  return f_add(\n    1,\n    f_sub(\n      2,\n      3\n    )\n  )"
            ),
        ],
    )
    def test_str(self, setup, typed, tree, expected):
        """
        Test that str method returns a string representation of the tree
        """
        tree_ind = self.typed_tree if typed else self.untyped_tree
        tree_ind.tree = tree

        tree_str = str(tree_ind)
        assert tree_str == expected

    @pytest.mark.parametrize(
        "typed, tree, root_indices, expected_results",
        [
            (
                True,
                [
                    FunctionNode(add2floats),
                    TerminalNode(1.0, float),
                    FunctionNode(sub2floats),
                    TerminalNode(2.0, float),
                    TerminalNode(3.0, float),
                ],
                [0, 1, 2, 3, 4],
                [
                    [
                        FunctionNode(add2floats),
                        TerminalNode(1.0, float),
                        FunctionNode(sub2floats),
                        TerminalNode(2.0, float),
                        TerminalNode(3.0, float),
                    ],
                    [TerminalNode(1.0, float)],
                    [
                        FunctionNode(sub2floats),
                        TerminalNode(2.0, float),
                        TerminalNode(3.0, float),
                    ],
                    [TerminalNode(2.0, float)],
                    [TerminalNode(3.0, float)],
                ],
            ),
            (
                False,
                [
                    FunctionNode(f_add),
                    TerminalNode(1),
                    FunctionNode(f_sub),
                    TerminalNode(2),
                    TerminalNode(3),
                ],
                [0, 1, 2, 3, 4],
                [
                    [
                        FunctionNode(f_add),
                        TerminalNode(1),
                        FunctionNode(f_sub),
                        TerminalNode(2),
                        TerminalNode(3),
                    ],
                    [TerminalNode(1)],
                    [
                        FunctionNode(f_sub),
                        TerminalNode(2),
                        TerminalNode(3),
                    ],
                    [TerminalNode(2)],
                    [TerminalNode(3)],
                ],
            ),
        ],
    )
    def test_get_subtree_by_root(
        self,
        setup,
        typed,
        tree,
        root_indices: List[int],
        expected_results: List[List[TreeNode]],
    ):
        tree_ind = self.typed_tree if typed else self.untyped_tree
        tree_ind.tree = tree

        for root_idx, expected in zip(root_indices, expected_results):
            subtree_root = tree[root_idx]
            assert tree_ind._get_subtree_by_root(subtree_root) == expected

    @pytest.mark.parametrize(
        "typed",
        [True, False]
    )
    def test_random_function(
        self,
        typed: bool,
    ):
        tree = self.typed_tree if typed else self.untyped_tree
        function_set = (
            self.typed_functions
            if typed
            else self.untyped_functions
        )
        node_type = float if typed else None

        for i in range(10):
            function_node = tree.random_function(node_type)
            assert function_node.function in function_set

        class MyType:
            pass

        assert tree.random_function(MyType) is None

    @pytest.mark.parametrize(
        "typed",
        [True, False]
    )
    def test_random_terminal(
        self,
        typed: bool,
    ):
        tree = self.typed_tree if typed else self.untyped_tree
        terminal_set = (
            self.typed_terminals
            if typed
            else self.untyped_terminals
        )
        node_type = float if typed else None

        for i in range(10):
            terminal = tree.random_terminal(node_type)
            assert terminal.value in terminal_set

        class MyType:
            pass

        assert tree.random_terminal(MyType) is None

    @pytest.mark.parametrize(
        "kwargs, error_message_substring",
        [
            # supplying int terminal with boolean functions only
            (
                {
                    "function_set": [and2bools, or2bools],
                    "terminal_set": {1: int, False: bool},
                    "root_type": bool
                },
                "must match terminal types"
            ),
            # supplying root_type int with no functions that return int
            (
                {
                    "function_set": [and2bools, or2bools],
                    "terminal_set": {True: bool, False: bool},
                    "root_type": int
                },
                "Root type must be the return type of at least one function."
            ),
            # supplying erc_range with no numeric functions
            (
                {
                    "erc_range": (1, 2),
                    "function_set": [and2bools, or2bools],
                    "terminal_set": {True: bool, False: bool},
                    "root_type": bool
                },
                "ERC range should not be defined if there are no numeric functions."
            ),
        ],
    )
    def test_init_bad_args(self, kwargs, error_message_substring):
        with pytest.raises(ValueError) as e:
            Tree(**kwargs)
        assert error_message_substring in str(e.value)

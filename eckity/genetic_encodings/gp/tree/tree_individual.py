"""
This module implements the tree class.
"""

import logging
import random
from numbers import Number
from typing import Callable, List, Union

import numpy as np

from eckity.base.utils import arity
from eckity.fitness.fitness import Fitness
from eckity.genetic_encodings.gp.tree.functions import (
    f_add,
    f_div,
    f_mul,
    f_sub,
)
from eckity.genetic_encodings.gp.tree.tree_node import TreeNode
from eckity.genetic_encodings.gp.tree.utils import _generate_args
from eckity.individual import Individual

logger = logging.getLogger(__name__)


class Tree(Individual):
    """
    A tree optimized for genetic programming operations.
    It is represented by a list of nodes in depth-first order.
    There are two types of nodes: functions and terminals.

    (tree is not meant as a stand-alone,
    parameters are supplied through the call from the Tree Creators)

    Parameters
    ----------
    init_depth: (int, int), default=None
        Min and max depths of initial random trees.

    function_set: list, default=None
        List of functions used as internal nodes in the GP tree.

    terminal_set: list, default=None
        List of terminals used in the GP-tree leaves.
    """

    def __init__(
        self,
        fitness: Fitness,
        function_set: List[Callable] = None,
        terminal_set: List[Union[str, Number]] = None,
        init_depth=(1, 2),
    ):
        super().__init__(fitness)
        if function_set is None:
            function_set = [f_add, f_sub, f_mul, f_div]

        if terminal_set is None:
            terminal_set = ["x", "y", "z", 0, 1, -1]

        self.function_set = function_set
        self.terminal_set = terminal_set
        self.arity = {func: arity(func) for func in self.function_set}
        self.vars = [t for t in terminal_set if not isinstance(t, Number)]
        self.init_depth = init_depth

        self.root: TreeNode = None  # actual tree representation

    @property
    def tree(self) -> TreeNode:
        logger.warn(
            "Tree.tree is deprecated in version 0.2 and will be \
                removed in version 0.3. Please use Tree.root instead."
        )

    def size(self):
        """
        Compute size of tree.

        Returns
        -------
        int
            tree size (= number of nodes).
        """
        return len(self.tree)

    def add_child(self, node, parent=None):
        if self.root is None:
            self.root = node
        else:
            parent.add_child(node)

    def add_tree(self, node, parent=None):
        logger.warn(
            "Tree.add_tree is deprecated and will be removed in\
                     version 0.3. Please use Tree.add_child instead."
        )
        return self.add_child(node, parent)

    def empty_tree(self):
        self.root = None

    def depth(self):
        # TODO update implementation to be TreeNode compatible

        """
        Compute depth of tree (maximal path length to a leaf).

        Returns
        -------
        int
            tree depth.
        """
        if self.root is None:
            return 0
        return self.root.depth(0)

    def random_function(self):
        """select a random function"""
        return random.choice(self.function_set)

    def random_terminal(self):
        """Select a random terminal"""
        return random.choice(self.terminal_set)

    def execute(self, *args, **kwargs):
        """
        Execute the program (tree).
        Input is a numpy array or keyword arguments (but not both).

        Parameters
        ----------
        args : arguments
            A numpy array.

        kwargs : keyword arguments
            Input to program, including every variable in the terminal set as a keyword argument.
            For example, if `terminal_set=['x', 'y', 'z', 0, 1, -1]`
            then call `execute(x=..., y=..., z=...)`.

        Returns
        -------
        object
            Result of tree execution.
        """

        if self.root is None:
            raise ValueError("Tree is empty, cannot execute.")

        reshape = False
        if args != ():  # numpy array -- convert to kwargs
            try:
                X = args[0]
                kwargs = _generate_args(X)
                reshape = True
            except Exception:
                raise ValueError(
                    f"Bad argument to tree.execute, must be numpy array or kwargs: {args}"
                )

        kw = list(kwargs.keys())

        bad_vars = [item for item in kw if item not in self.vars]
        if len(bad_vars) > 0:
            raise ValueError(
                f"tree.execute received variable arguments not in terminal set: {bad_vars}"
            )

        missing_vars = [item for item in self.vars if item not in kw]
        if len(missing_vars) > 0:
            raise ValueError(
                f"Some variable terminals were not passed to tree.execute as keyword arguments: {missing_vars}"
            )

        res = self.root.execute(**kwargs)
        if reshape and (isinstance(res, Number) or res.shape == np.shape(0)):
            # sometimes a tree degenrates to a scalar value
            res = np.full_like(X[:, 0], res)
        return res

    def random_subtree(self, node_type=None):
        relevant_nodes = self.root.filter_by_type(node_type, [])
        return random.choice(relevant_nodes)

    def replace_subtree(self, old_subtree: TreeNode, new_subtree: TreeNode):
        if self.root == old_subtree:
            self.root = new_subtree
        else:
            self.root.replace_child(old_subtree, new_subtree)

    def __str__(self):
        result = [f"def func_{self.id}({', '.join(self.terminal_set)}):\n   return "]
        self.root.generate_tree_code("   ", result)
        return "".join(result)

    def show(self):
        """
        Print out a simple textual representation of the tree.

        Returns
        -------
        None.
        """
        logger.info("\n" + str(self))


# end class tree

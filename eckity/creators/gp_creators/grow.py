from random import random
from types import NoneType

from overrides import overrides

from eckity.creators.gp_creators.tree_creator import GPTreeCreator
from eckity.genetic_encodings.gp import (
    FunctionNode,
    TerminalNode,
    Tree,
    TreeNode,
)


class GrowCreator(GPTreeCreator):
    def __init__(
        self,
        init_depth=None,
        function_set=None,
        terminal_set=None,
        bloat_weight=0.0,
        events=None,
    ):
        """
        Tree creator using the grow method

        Parameters
        ----------
        init_depth : (int, int)
        Min and max depths of initial random trees. The default is None.

        function_set : list
                List of functions used as internal nodes in the GP-tree. The default is None.

        terminal_set : list
                List of terminals used in the GP-tree leaves. The default is None.

        bloat_weight : float
                Bloat control weight to punish large trees. Bigger values make a bigger punish.

        events : list
                List of events related to this class
        """
        super().__init__(
            init_depth=init_depth,
            function_set=function_set,
            terminal_set=terminal_set,
            bloat_weight=bloat_weight,
            events=events,
        )

    @overrides
    def create_tree(self, tree_ind):
        """
        Create a random tree using the grow method, and assign it to the given individual.

        Parameters
        ----------
        tree_ind: Tree
                An individual of GP Tree representation with an initially empty tree

        Returns
        -------
        None.
        """
        root = self.build_tree(tree_ind, depth=0)
        tree_ind.root = root

    def build_tree(
        self,
        tree_ind: Tree,
        depth: int,
        node_type: type = NoneType,
        parent: TreeNode = None,
    ) -> TreeNode:
        """
        Recursively create a random tree using the grow method

        Parameters
        ----------
        depth: int
                Current depth in recursive process.

        Returns
        -------
        None.

        """
        is_func = False
        min_depth, max_depth = self.init_depth

        if depth < min_depth:
            node = tree_ind.random_function_node(
                node_type=node_type, parent=parent
            )
            is_func = True
        elif depth >= max_depth:
            node = tree_ind.random_terminal_node(
                node_type=node_type, parent=parent
            )
        else:  # intermediate depth, grow
            if random() > 0.5:
                node = tree_ind.random_function_node(
                    node_type=node_type, parent=parent
                )
                is_func = True
            else:
                node = tree_ind.random_terminal_node(
                    node_type=node_type, parent=parent
                )

        if is_func:
            # recursively add children to the function node
            func_types = FunctionNode.get_func_types(node.function)
            for i in range(node.n_children):
                child_node = self.build_tree(
                    tree_ind,
                    depth=depth + 1,
                    node_type=func_types[i],
                    parent=node,
                )
                node.add_child(child_node)

        return node

from random import random

from overrides import overrides

from eckity.creators.gp_creators.tree_creator import GPTreeCreator


class GrowCreator(GPTreeCreator):
    def __init__(
        self,
        init_depth=None,
        function_set=None,
        terminal_set=None,
        bloat_weight=0.1,
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
        self.create_tree_rec(tree_ind, 0)

    def create_tree_rec(self, tree_ind, depth, parent=None):
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
            node = tree_ind.random_function_node(parent=parent)
            is_func = True
        elif depth >= max_depth:
            node = tree_ind.random_terminal_node(parent=parent)
        else:  # intermediate depth, grow
            if random() > 0.5:
                node = tree_ind.random_function_node(parent=parent)
                is_func = True
            else:
                node = tree_ind.random_terminal_node(parent=parent)

        # add the new node to the tree of the given individual
        tree_ind.add_child(node, parent=parent)

        if is_func:
            # recursively add children to the function node
            for i in range(node.n_children):
                self.create_tree_rec(tree_ind, depth=depth + 1, parent=node)

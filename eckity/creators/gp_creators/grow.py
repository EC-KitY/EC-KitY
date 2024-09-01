import random
from types import NoneType
from typing import Any, Callable, Dict, List, Tuple, Union

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
        init_depth: Tuple[int, int] = None,
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        bloat_weight: float = 0.0,
        p_prune: float = 0.5,
        events: List[str] = None,
        root_type: type = NoneType,
    ):
        """
        Tree creator using the grow method

        Parameters
        ----------
        init_depth : (int, int)
        Min and max depths of initial random trees. The default is None.

        function_set : list, default=None
                List of functions used as internal nodes in the GP-tree.

        terminal_set : list, default=None.
                List of terminals used in the GP-tree leaves.

        bloat_weight : float, default=0.0
                Bloat control weight to punish large trees.

        p_prune : float, default=0.5
                Probability of pruning the tree at each level.

        events : list
                List of events related to this class
        """
        super().__init__(
            init_depth=init_depth,
            function_set=function_set,
            terminal_set=terminal_set,
            bloat_weight=bloat_weight,
            events=events,
            root_type=root_type,
        )
        self.p_prune = p_prune

    @overrides
    def create_tree(
        self,
        tree: List[TreeNode],
        random_function: Callable[type, FunctionNode],
        random_terminal: Callable[type, TerminalNode],
        depth: int = 0,
        node_type: type = NoneType,
    ) -> None:
        """
        Recursively create a random tree using the grow method.

        Parameters
        ----------
        tree_ind : Tree
            Tree Individual that will receive the generated tree.
        depth : int
            Current depth in recursive process.
        node_type : type, optional
            Type of the node to create. The default is NoneType.
        """
        if tree is None:
            tree = []

        min_depth, max_depth = self.init_depth

        is_func = False
        if depth < min_depth:
            node = random_function(node_type)
            is_func = True
        elif depth >= max_depth:
            node = random_terminal(node_type)
        else:  # intermediate depth, grow
            if random.random() < self.p_prune:
                node = random_terminal(node_type)
            else:
                node = random_function(node_type)
                is_func = True

        # add the new node to the tree of the given individual
        tree.append(node)

        if is_func:
            self._add_children(
                tree=tree,
                fn_node=node,
                random_function=random_function,
                random_terminal=random_terminal,
                depth=depth,
            )

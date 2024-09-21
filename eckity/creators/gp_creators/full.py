from types import NoneType
from typing import Any, Callable, Dict, List, Tuple, Union

from overrides import overrides

from eckity.creators.gp_creators.tree_creator import GPTreeCreator
from eckity.genetic_encodings.gp import (
    Tree,
    TreeNode,
    TerminalNode,
    FunctionNode,
)
from eckity.genetic_encodings.gp.tree.utils import get_func_types


class FullCreator(GPTreeCreator):
    def __init__(
        self,
        init_depth: Tuple[int, int] = None,
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        bloat_weight: float = 0.0,
        events: List[str] = None,
        root_type: type = NoneType,
        update_parents: bool = False,
    ):
        """
        Tree creator using the full method

        Parameters
        ----------
        init_depth : (int, int)
        Min and max depths of initial random trees. The default is None.

        function_set : list
                List of functions used as internal nodes in the GP tree. The default is None.

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
            root_type=root_type,
            update_parents=update_parents,
        )

    @overrides
    def create_tree(
        self,
        tree,
        random_function: Callable[type, FunctionNode],
        random_terminal: Callable[type, TerminalNode],
        depth: int = 0,
        node_type: type = NoneType,
    ) -> None:
        """
        Recursively create a random tree using the full method

        Parameters
        ----------
        depth: int
                Current depth in recursive process.

        Returns
        -------
        None.

        """
        max_depth = self.init_depth[1]

        if depth >= max_depth:
            node = random_terminal(node_type)

            # add the new node to the tree of the given individual
            tree.append(node)
        else:
            node = random_function(node_type)

            # add the new node to the tree of the given individual
            tree.append(node)

            # recursively add argument nodes to the tree
            func_types = get_func_types(node.function)[:-1]
            for t in func_types:
                self.create_tree(
                    tree,
                    random_function,
                    random_terminal,
                    depth=depth + 1,
                    node_type=t,
                )

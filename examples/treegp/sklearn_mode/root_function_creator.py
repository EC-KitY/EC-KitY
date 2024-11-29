from typing import Callable, List, Optional

from overrides import override

from eckity.creators import FullCreator
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, TreeNode


class RootFunctionCreator(FullCreator):
    """
    Creator for trees with an immutable root function
    (e.g. argmax/softmax for classification)
    """

    def __init__(
        self,
        root_function: Callable = None,
        init_depth=None,
        function_set=None,
        terminal_set=None,
        bloat_weight=0.1,
        events=None,
    ) -> None:
        super().__init__(
            init_depth=init_depth,
            function_set=function_set,
            terminal_set=terminal_set,
            bloat_weight=bloat_weight,
            events=events,
        )
        self.root_function = root_function

    @override
    def create_tree(
        self,
        tree: List[TreeNode],
        random_function: Callable[[type], Optional[FunctionNode]],
        random_terminal: Callable[[type], Optional[TerminalNode]],
        depth: int = 0,
        node_type: Optional[type] = None,
    ) -> None:
        root = FunctionNode(self.root_function)

        self._add_children(
            tree=tree,
            fn_node=root,
            random_function=random_function,
            random_terminal=random_terminal,
            depth=0,
        )

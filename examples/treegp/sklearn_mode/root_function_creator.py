from eckity.creators import FullCreator
from typing import Callable
from overrides import override
from eckity.genetic_encodings.gp import FunctionNode


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
    def create_tree(self, tree_ind):
        root = FunctionNode(self.root_function)
        tree_ind.root = root

        self._build_children(root, tree_ind, depth=0)

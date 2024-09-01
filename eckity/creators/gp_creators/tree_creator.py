from types import NoneType
from typing import Any, Callable, Dict, List, Tuple, Union

from overrides import override

from eckity.base.utils import arity
from eckity.creators.creator import Creator
from eckity.fitness.gp_fitness import GPFitness
from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.gp import (
    FunctionNode,
    TerminalNode,
    Tree,
    TreeNode,
)
from eckity.genetic_encodings.gp.tree.utils import get_func_types


class GPTreeCreator(Creator):
    def __init__(
        self,
        init_depth: Tuple[int, int] = None,
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        fitness_type: type = SimpleFitness,
        bloat_weight: float = 0.0,
        events: List[str] = None,
        root_type: type = NoneType,
    ):
        if events is None:
            events = ["after_creation"]
        super().__init__(events, fitness_type)

        if init_depth is None:
            init_depth = (2, 4)

        if function_set is None:
            raise ValueError("function_set must be provided")

        if terminal_set is None:
            raise ValueError("terminal_set must be provided")

        self.init_depth = init_depth
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.bloat_weight = bloat_weight
        self.root_type = root_type

    @override
    def create_individuals(
        self, n_individuals: int, higher_is_better: bool
    ) -> List[Tree]:
        individuals = [
            Tree(
                function_set=self.function_set,
                terminal_set=self.terminal_set,
                fitness=GPFitness(
                    bloat_weight=self.bloat_weight,
                    higher_is_better=higher_is_better,
                ),
                root_type=self.root_type,
            )
            for _ in range(n_individuals)
        ]
        for ind in individuals:
            self.create_tree(
                ind.tree,
                ind.random_function,
                ind.random_terminal,
                node_type=self.root_type,
            )
        self.created_individuals = individuals
        return individuals

    def create_tree(
        self,
        tree: List[TreeNode],
        random_function: Callable[type, FunctionNode],
        random_terminal: Callable[type, TerminalNode],
        depth: int = 0,
        node_type: type = NoneType,
    ) -> None:
        """
        Create the actual tree representation of an existing Tree individual

        Parameters
        ----------
        tree_ind : Tree
            Individual to create the tree representation for
        """
        pass

    def _add_children(
        self,
        tree: List[TreeNode],
        fn_node: FunctionNode,
        random_function: Callable[type, FunctionNode],
        random_terminal: Callable[type, TerminalNode],
        depth: int,
    ) -> None:
        # recursively add children to the function node
        func_types = get_func_types(fn_node.function)
        for i in range(arity(fn_node.function)):
            self.create_tree(
                tree,
                random_function,
                random_terminal,
                depth=depth + 1,
                node_type=func_types[i],
            )

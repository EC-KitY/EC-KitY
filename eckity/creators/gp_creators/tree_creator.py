from typing import Any, Callable, Dict, List, Tuple, Union, Optional

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
        erc_range: Union[Tuple[int, int], Tuple[float, float]] = None,
        fitness_type: type = SimpleFitness,
        bloat_weight: float = 0.0,
        events: List[str] = None,
        root_type: Optional[type] = None,
        update_parents: bool = False
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
        self.update_parents = update_parents
        self.erc_range = erc_range

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
                update_parents=self.update_parents,
                erc_range=self.erc_range,
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
        random_function: Callable[[type], Optional[FunctionNode]],
        random_terminal: Callable[[type], Optional[TerminalNode]],
        depth: int = 0,
        node_type: Optional[type] = None,
    ) -> None:
        """
        Recursively build the tree representation
        of an existing Tree individual.
        
        (This method is not abstract as it is not required in HalfCreator)

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
        random_function: Callable[[type], Optional[FunctionNode]],
        random_terminal: Callable[[type], Optional[TerminalNode]],
        depth: int,
    ) -> None:
        """
        Recursively add children to a function node.

        Parameters
        ----------
        tree : List[TreeNode]
            Total representation of the tree.
        fn_node : FunctionNode
            Function node, parent of the children to be created.
        random_function : Callable[[type], Optional[FunctionNode]]
            Random FunctionNode generator.
        random_terminal : Callable[[type], Optional[TerminalNode]]
            Random TerminalNode generator.
        depth : int
            current depth of the tree
        """
        func_types = get_func_types(fn_node.function)
        for i in range(arity(fn_node.function)):
            self.create_tree(
                tree,
                random_function,
                random_terminal,
                depth=depth + 1,
                node_type=func_types[i],
            )

    def _assert_node_created(
        self,
        node: Optional[TreeNode],
        node_type: Optional[type] = None
    ) -> None:
        """
        Assert that a TreeNode was created successfully.

        Parameters
        ----------
        node : Optional[TreeNode]
            Generated tree node.
        node_type : Optional[type], optional
            Generated tree node type, by default None

        Raises
        ------
        ValueError
            Raised if a node was not generated successfully.
        """
        if node is None:
            # optionally add type info for typed case
            type_info = (
                f"with type {node_type}" if node_type is not None else ""
            )

            # optionally add erc_range info if it was defined
            erc_info = (
                f"erc range ({self.erc_range}) ,"
                if self.erc_range is not None
                else ""
            )

            raise ValueError(
                f"Could not generate node {type_info} "
                f"due to a mismatch in terminals ({self.terminal_set}), "
                f"functions ({self.function_set}), {erc_info} "
                f"or init_depth ({self.init_depth}) configuration."
            )

from typing import Any, Callable, Dict, List, Tuple, Union

from overrides import override

from eckity.base.untyped_functions import f_add, f_div, f_mul, f_sub
from eckity.creators.creator import Creator
from eckity.fitness.gp_fitness import GPFitness
from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.gp.tree.tree_individual import FunctionNode, Tree


class GPTreeCreator(Creator):
    def __init__(
        self,
        init_depth: Tuple[int, int] = None,
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        fitness_type: type = SimpleFitness,
        bloat_weight: float = 0.0,
        events: List[str] = None,
    ):
        if events is None:
            events = ["after_creation"]
        super().__init__(events, fitness_type)

        if init_depth is None:
            init_depth = (2, 4)

        if function_set is None:
            function_set = [f_add, f_sub, f_mul, f_div]

        if terminal_set is None:
            terminal_set = ["x", "y", "z", 0, 1, -1]

        self.init_depth = init_depth
        self.function_set = function_set
        self.terminal_set = terminal_set
        self.bloat_weight = bloat_weight

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
                init_depth=self.init_depth,
            )
            for _ in range(n_individuals)
        ]
        for ind in individuals:
            self.create_tree(ind)
        self.created_individuals = individuals
        return individuals

    def create_tree(self, tree_ind: Tree) -> None:
        """
        Create the actual tree representation of an existing Tree individual

        Parameters
        ----------
        tree_ind : Tree
            Individual to create the tree representation for
        """
        pass

    def _build_children(
        self, node: FunctionNode, tree_ind: Tree, depth: int
    ) -> None:
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

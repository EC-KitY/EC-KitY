from typing import Any, List, Optional, Tuple

from overrides import override

from eckity.creators.gp_creators.grow import GrowCreator
from eckity.genetic_encodings.gp import Tree, TreeNode
from eckity.genetic_operators import FailableOperator


class SubtreeMutation(FailableOperator):
    def __init__(
        self,
        arity=1,
        probability: float = 1.0,
        init_depth: Tuple[int, int] = (2, 4),
        events=None,
        attempts=1,
    ):
        super().__init__(
            probability=probability, arity=1, events=events, attempts=attempts
        )
        self.init_depth = init_depth
        self.tree_creator = None

    @override
    def attempt_operator(
        self, payload: Any, attempt_num: int
    ) -> Tuple[bool, Any]:
        """
        Perform subtree mutation: select a subtree at random
        to be replaced by a new, randomly generated subtree.

        Returns
        -------
        Tuple[bool, Any]
            A tuple containing a boolean indicating whether the operator was
            successful and a list of the individuals.
        """
        individuals: List[Tree] = payload

        old_subtrees: List[Optional[List[TreeNode]]] = [
            ind.random_subtree() for ind in individuals
        ]

        if None in old_subtrees:
            return False, individuals

        self._swap_subtrees(individuals, old_subtrees)

        self.applied_individuals = individuals
        return True, individuals

    def _swap_subtrees(
        self, individuals: List[Tree], old_subtrees: List[List[TreeNode]]
    ) -> None:

        if self.tree_creator is None:
            self.tree_creator = GrowCreator(
                init_depth=self.init_depth,
                function_set=individuals[0].function_set,
                terminal_set=individuals[0].terminal_set,
            )

        for ind, old_subtree in zip(individuals, old_subtrees):
            # generate a random tree with the same root type
            # of the old subtree to not cause type errors

            new_subtree = []
            self.tree_creator.create_tree(
                new_subtree,
                ind.random_function,
                ind.random_terminal,
                depth=0,
                node_type=old_subtree[0].node_type,
            )

            # replace the old subtree with the newly generated one
            ind.replace_subtree(
                old_subtree=old_subtree, new_subtree=new_subtree
            )

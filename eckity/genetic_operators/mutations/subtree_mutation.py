from types import NoneType
from typing import Any, List, Tuple

from overrides import override

from eckity.creators.gp_creators.grow import GrowCreator
from eckity.genetic_encodings.gp import Tree, TreeNode
from eckity.genetic_operators import FailableOperator


class SubtreeMutation(FailableOperator):
    def __init__(
        self,
        node_type=NoneType,
        probability=1,
        arity=1,
        init_depth=(2, 4),
        events=None,
    ):
        super().__init__(probability=probability, arity=arity, events=events)
        self.node_type = node_type
        self.init_depth = init_depth

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
        old_subtrees: List[TreeNode] = [
            ind.random_subtree(node_type=self.node_type) for ind in individuals
        ]

        # Failed attempt
        if None in old_subtrees:
            return False, individuals

        tree_creator = GrowCreator(
            init_depth=self.init_depth,
            function_set=individuals[0].function_set,
            terminal_set=individuals[0].terminal_set,
        )

        for ind, old_subtree in zip(individuals, old_subtrees):
            new_subtree = tree_creator.create_tree(
                ind,
                depth=0,
                node_type=old_subtree[0].node_type
            )
            ind.replace_subtree(
                old_subtree=old_subtree, new_subtree=new_subtree
            )

        self.applied_individuals = individuals
        return True, individuals

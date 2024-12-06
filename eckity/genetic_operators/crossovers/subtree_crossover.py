
from typing import Any, List, Tuple, Optional

from overrides import override

from eckity.genetic_encodings.gp import Tree, TreeNode
from eckity.genetic_operators.failable_operator import FailableOperator


class SubtreeCrossover(FailableOperator):
    def __init__(self, probability=1.0, arity=2, events=None, attempts=1):
        super().__init__(
            probability=probability,
            arity=arity,
            events=events,
            attempts=attempts,
        )
        self.individuals = None
        self.applied_individuals = None

    @override
    def attempt_operator(
        self, payload: Any, attempt_num: int
    ) -> Tuple[bool, Any]:
        """
        Perform subtree crossover between a list of trees in a cyclic manner.
        Meaning, the second individual will have a subtree from the first,
        and the first individual will have a subtree from the last individual.

        Parameters
        ----------
        payload: List[Individual]
            List of Trees to perform crossover on

        individual: Tree
        tree individual to perform crossover on

        Returns
        -------
        List
            List of individuals after crossover (modified in-place)
        """
        individuals = payload

        if len(individuals) != self.arity:
            raise ValueError(
                f"Expected individuals of size {self.arity}, "
                f"got {len(individuals)}."
            )

        self.individuals = individuals

        subtrees: Optional[List[List[TreeNode]]] = self._pick_subtrees(
            individuals
        )

        if subtrees is None:
            return False, individuals

        self._swap_subtrees(individuals, subtrees)
        self.applied_individuals = individuals

        return True, individuals

    @staticmethod
    def _pick_subtrees(
        individuals: List[Tree],
    ) -> Optional[List[List[TreeNode]]]:
        # select a random subtree from first individual tree
        first_subtree: List[TreeNode] = individuals[0].random_subtree()

        if first_subtree is None:
            # failed attempt
            return None

        m_type: type = first_subtree[0].node_type

        # now select a random subtree from the rest of the individuals
        # with regards to the type of the first subtree
        rest_subtrees = [ind.random_subtree(m_type) for ind in individuals[1:]]

        # fails if any subtree doesn't contain a node with of type `m_type`
        if None in rest_subtrees:
            return None

        subtrees = [first_subtree] + rest_subtrees
        return subtrees

    @staticmethod
    def _swap_subtrees(
        individuals: List[TreeNode], subtrees: List[List[TreeNode]]
    ) -> None:
        """
        Replace subtrees for all individuals in a cyclic manner
        For n subtrees (st_1, st_2, ..., st_n):
        st_n receives the subtree of st_n-1
        st_n-1 receives the subtree of st_n-2
        ...
        st_2 receives the subtree of st_1
        st_1 receives the subtree of st_n
        """
        for i in range(len(individuals) - 1, -1, -1):
            individuals[i].replace_subtree(
                old_subtree=subtrees[i], new_subtree=subtrees[i - 1]
            )

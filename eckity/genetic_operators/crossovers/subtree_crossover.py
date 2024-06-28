from types import NoneType
from typing import Any, Tuple

from overrides import override

from eckity.genetic_operators.failable_operator import FailableOperator


class SubtreeCrossover(FailableOperator):
    def __init__(
        self, node_type=NoneType, probability=1, arity=2, events=None
    ):
        super().__init__(probability=probability, arity=arity, events=events)
        self.individuals = None
        self.applied_individuals = None
        self.node_type = node_type

    # TODO add type hints
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

        # select a random subtree from each individual's tree
        subtrees = [ind.random_subtree(self.node_type) for ind in individuals]

        if None in subtrees:
            # failed attempt
            return False, individuals

        # replace subtrees for all individuals in a cyclic manner
        for i in range(len(individuals) - 1):
            individuals[i].replace_subtree(
                old_subtree=subtrees[i], new_subtree=subtrees[i + 1]
            )
        individuals[-1].replace_subtree(
            old_subtree=subtrees[-1], new_subtree=subtrees[0]
        )

        return True, individuals

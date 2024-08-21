from types import NoneType
from typing import Any, Tuple

from overrides import override

from eckity.genetic_operators.failable_operator import FailableOperator


class SubtreeCrossover(FailableOperator):
    def __init__(self, probability=1.0, arity=2, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
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

        # all individuals should have the same terminal_set
        # so it doesn't matter which individual is invoked here
        m_type = individuals[0].random_type()

        # select a random subtree from each individual's tree
        subtrees = [
            ind.random_subtree(m_type)
            for ind in individuals
        ]

        if None in subtrees:
            # failed attempt
            return False, individuals

        # Replace subtrees for all individuals in a cyclic manner
        # For n subtrees (st_1, st_2, ..., st_n):
        # st_n receives the subtree of st_n-1
        # st_n-1 receives the subtree of st_n-2
        # ...
        # st_1 receives the subtree of st_0
        # st_0 receives the subtree of st_n
        for i in range(len(individuals) - 1, -1, -1):
            individuals[i].replace_subtree(
                old_subtree=subtrees[i], new_subtree=subtrees[i - 1]
            )

        return True, individuals

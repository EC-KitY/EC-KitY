from typing import Any, List, Tuple

from overrides import override
import random

from eckity.genetic_encodings.gp import Tree
from eckity.genetic_operators import FailableOperator


class ERCMutation(FailableOperator):
    def __init__(
        self,
        probability=1.0,
        arity=1,
        mu=0,
        sigma=1,
        events=None,
    ):
        super().__init__(probability=probability, arity=arity, events=events)
        self.mu = mu
        self.sigma = sigma

    @override
    def attempt_operator(
        self, payload: Any, attempt_num: int
    ) -> Tuple[bool, Any]:
        """
        Perform ephemeral random constant (ERC) mutation: select an ERC node at random
        and add Gaussian noise to it.

        Returns
        -------
        None.
        """
        individuals: List[Tree] = payload

        leaves = [ind.get_random_numeric_node() for ind in individuals]

        if None in leaves:
            return False, individuals

        mu, sigma = self.mu, self.sigma
        for terminal in leaves:
            terminal.value = terminal.value + random.gauss(mu, sigma)

        self.applied_individuals = individuals
        return True, individuals

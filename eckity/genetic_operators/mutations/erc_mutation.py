from random import choice, gauss

from eckity.event_based_operator import Operator
from eckity.genetic_operators.genetic_operator import GeneticOperator


class ERCMutation(GeneticOperator):
    def __init__(self, probability=1, arity=1, events=None):
        super().__init__(probability=probability, arity=arity, events=events)

    def apply(self, individuals):
        """
        Perform ephemeral random constant (ERC) mutation: select an ERC node at random
        and add Gaussian noise to it.

        Returns
        -------
        None.
        """
        for j in range(len(individuals)):
            erc_indexes = [i for i, node in enumerate(individuals[j].tree) if
                           isinstance(node, (int, float)) and node not in individuals[j].terminal_set]
            if len(erc_indexes) > 0:
                m_point = choice(erc_indexes)
                individuals[j].tree[m_point] = round(individuals[j].tree[m_point] + gauss(mu=0, sigma=1), 4)
        self.applied_individuals = individuals
        return individuals

from random import gauss

from eckity.genetic_operators.genetic_operator import GeneticOperator


class ERCMutation(GeneticOperator):
    def __init__(
        self,
        probability=1.0,
        arity=1,
        erc_range=(-5, 5),
        mu=0,
        sigma=1,
        events=None,
    ):
        super().__init__(probability=probability, arity=arity, events=events)
        self.erc_range = erc_range
        self.mu = mu
        self.sigma = sigma

    def apply(self, individuals):
        """
        Perform ephemeral random constant (ERC) mutation: select an ERC node at random
        and add Gaussian noise to it.

        Returns
        -------
        None.
        """
        mu, sigma = self.mu, self.sigma
        for ind in individuals:
            subtree = ind.random_subtree(node_type=float)
            subtree.value = subtree.value + gauss(mu, sigma)

        self.applied_individuals = individuals
        return individuals

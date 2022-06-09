from eckity.genetic_operators.genetic_operator import GeneticOperator


class SubtreeCrossover(GeneticOperator):
    def __init__(self, probability=1, arity=2, events=None):
        self.individuals = None
        self.applied_individuals = None
        super().__init__(probability=probability, arity=arity, events=events)

    def apply(self, individuals):
        """
        Perform subtree crossover between this tree and `other` tree:
            1. Select random node from `other` tree
            2. Get subtree rooted at selected node
            1. Select a random node in this tree
            2. Place `other` selected subtree at this node, replacing current subtree

        Parameters
        ----------
        individuals
        select_func: callable
        Selection method used to receive additional individuals to perform crossover on

        individual: Tree
        tree individual to perform crossover on

        Returns
        -------
        a new, modified individual
        """

        assert len(individuals) == self.arity, f'Expected individuals list of size {self.arity}, got {len(individuals)}'

        self.individuals = individuals

        # select a random subtree from each individual's tree
        subtrees = [ind.random_subtree() for ind in individuals]

        # assign the next individual's subtree to the current individual's tree
        for i in range(len(individuals) - 1):
            individuals[i].replace_subtree(subtrees[i+1])

        # to complete the crossover circle, assign the first subtree to the last individual
        individuals[-1].replace_subtree(subtrees[0])

        self.applied_individuals = individuals
        return individuals

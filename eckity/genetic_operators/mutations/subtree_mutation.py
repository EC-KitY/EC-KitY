from random import randint

from eckity.creators.gp_creators.grow import GrowCreator
from eckity.genetic_operators.genetic_operator import GeneticOperator


class SubtreeMutation(GeneticOperator):
    def __init__(self, probability=1, arity=1, init_depth=None, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        self.init_depth = init_depth

    def apply(self, individuals):
        """
        Perform subtree mutation: select a subtree at random to be replaced by a new, randomly generated subtree.

        Returns
        -------
        None.
        """

        for ind in individuals:
            # Select a random subtree
            subtree = ind.random_subtree()

            init_depth = (
                (
                    ind.init_depth[0],
                    randint(ind.init_depth[0], ind.init_depth[1]),
                )
                if self.init_depth is None
                else self.init_depth
            )
            tree_creator = GrowCreator(
                init_depth=init_depth,
                function_set=ind.function_set,
                terminal_set=ind.terminal_set,
            )

            tree_creator.create_tree_rec(ind, 0, parent=subtree.parent)
            ind.replace_subtree(old_subtree=subtree, new_subtree=ind.root)

        self.applied_individuals = individuals
        return individuals

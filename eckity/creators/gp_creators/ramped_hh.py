from eckity.creators.gp_creators.full import FullCreator
from eckity.creators.gp_creators.grow import GrowCreator
from eckity.creators.gp_creators.tree_creator import GPTreeCreator
from eckity.genetic_encodings.gp.tree.tree_individual import Tree
from eckity.fitness.gp_fitness import GPFitness


class RampedHalfAndHalfCreator(GPTreeCreator):
    def __init__(self,
                 grow_creator=None,
                 full_creator=None,
                 init_depth=None,
                 function_set=None,
                 terminal_set=None,
                 erc_range=None,
                 bloat_weight=0.1,
                 events=None):
        """
        Tree creator that creates trees using the Ramped Half and Half method

        Parameters
        ----------
        grow_creator: GrowCreator
            a tree creator that creates trees using the grow method

        full_creator: FullCreator
            a tree creator that creates trees using the grow method

        init_depth : (int, int)
        Min and max depths of initial random trees. The default is None.

        function_set : list
            List of functions used as internal nodes in the GP tree. The default is None.

        terminal_set : list
            List of terminals used in the GP-tree leaves. The default is None.

        erc_range : (float, float)
            Range of values for ephemeral random constant (ERC). The default is None.

        bloat_weight : float
            Bloat control weight to punish large trees. Bigger values make a bigger punish.

        events : list
            List of events related to this class
        """
        super().__init__(init_depth=init_depth,
                         function_set=function_set,
                         terminal_set=terminal_set,
                         erc_range=erc_range,
                         bloat_weight=bloat_weight,
                         events=events)

        # assign default creators
        if grow_creator is None:
            grow_creator = GrowCreator(init_depth=self.init_depth, function_set=self.function_set,
                                       terminal_set=self.terminal_set, erc_range=self.erc_range, events=self.events)
        if full_creator is None:
            full_creator = FullCreator(init_depth=self.init_depth, function_set=self.function_set,
                                       terminal_set=self.terminal_set, erc_range=self.erc_range, events=self.events)

        self.grow_creator, self.full_creator = grow_creator, full_creator

        self.init_method = None     # current creator in use (either grow or full)

    def create_individuals(self, n_individuals, higher_is_better):
        """
        Initialize the subpopulation individuals using ramped half-and-half method.

        Parameters
        ----------
        n_individuals: int
            number of individuals to create

        higher_is_better: bool
            determines if the fitness of the created individuals should be minimized or maximized

        Returns
        -------

        """

        min_depth, max_depth = self.init_depth[0], self.init_depth[1]

        # if pop size is 100 and we want depths 2,3,4,5,6 then group_size is 10:
        # 10 'grow' with depth 2, 10 'full' with depth 2, 10 'grow' with depth 3, 10 'full' with depth 3, etc.
        group_size = int(n_individuals / (max_depth + 1 - min_depth) / 2)

        individuals = []

        for depth in range(min_depth, max_depth + 1):
            for i in range(group_size):
                # as explained above, first create (group_size) individuals using grow method
                self.init_method = self.grow_creator
                self._create_individuals(individuals, depth, higher_is_better)

                # then create (group_size) individuals using full method
                self.init_method = self.full_creator
                self._create_individuals(individuals, depth, higher_is_better)

        # might need to add a few because 'group_size' may have been a float that was truncated
        self.init_method = self.full_creator
        for i in range(n_individuals - len(individuals)):
            self._create_individuals(individuals, max_depth, higher_is_better)

        # TODO we don't need this event since creators have before/after operator events
        # self.publish("after_creation")

        self.created_individuals = individuals
        return individuals

    def _create_individuals(self, individuals, max_depth, higher_is_better):
        t = Tree(init_depth=self.init_depth, function_set=self.function_set,
                 terminal_set=self.terminal_set, erc_range=self.erc_range,
                 fitness=GPFitness(bloat_weight=self.bloat_weight, higher_is_better=higher_is_better))
        self.create_tree(t, max_depth=max_depth)
        individuals.append(t)

    def create_tree(self, tree_ind, max_depth):
        self.init_method.create_tree(tree_ind, max_depth)

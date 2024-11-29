from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from overrides import override

from eckity.creators.gp_creators.full import FullCreator
from eckity.creators.gp_creators.grow import GrowCreator
from eckity.creators.gp_creators.tree_creator import GPTreeCreator
from eckity.genetic_encodings.gp.tree.tree_individual import Tree


class HalfCreator(GPTreeCreator):
    def __init__(
        self,
        grow_creator: GrowCreator = None,
        full_creator: FullCreator = None,
        init_depth: Tuple[int, int] = None,
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        bloat_weight: float = 0.0,
        erc_range: Union[Tuple[int, int], Tuple[float, float]] = None,
        events: List[str] = None,
        root_type: Optional[type] = None,
    ):
        """
        Tree creator that creates trees using the Ramped Half and Half method

        Parameters
        ----------
        grow_creator: GrowCreator
                a tree creator that creates trees using the grow method

        full_creator: FullCreator
                a tree creator that creates trees using the full method

        init_depth : (int, int)
        Min and max depths of initial random trees. The default is None.

        function_set : list
                List of functions used as internal nodes in the GP tree. The default is None.

        terminal_set : list or dict
                List of terminals used in the GP-tree leaves. The default is None.

        bloat_weight : float
                Bloat control weight to punish large trees. Bigger values make a bigger punish.

        events : list
                List of events related to this class
        """
        super().__init__(
            init_depth=init_depth,
            function_set=function_set,
            terminal_set=terminal_set,
            bloat_weight=bloat_weight,
            erc_range=erc_range,
            events=events,
            root_type=root_type,
        )

        # assign default creators
        if grow_creator is None:
            grow_creator = GrowCreator(
                init_depth=self.init_depth,
                function_set=self.function_set,
                terminal_set=self.terminal_set,
                events=self.events,
                root_type=root_type,
                erc_range=self.erc_range,
            )
        if full_creator is None:
            full_creator = FullCreator(
                init_depth=self.init_depth,
                function_set=self.function_set,
                terminal_set=self.terminal_set,
                events=self.events,
                root_type=root_type,
                erc_range=self.erc_range,
            )

        self.grow_creator = grow_creator
        self.full_creator = full_creator

    @override
    def create_individuals(
        self, n_individuals: int, higher_is_better: bool
    ) -> List[Tree]:
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
        min_depth, max_depth = self.init_depth

        # if pop size is 100 and we want depths 2,3,4,5,6 then group_size is 10:
        # 10 'grow' with depth 2, 10 'full' with depth 2, 10 'grow' with depth 3, 10 'full' with depth 3, etc.
        group_size = int(n_individuals / (2 * (max_depth + 1 - min_depth)))

        if group_size == 0:
            raise ValueError(
                "Incompatible population size and init_depth. "
                "Population size must be "
                f"at least {2 * (max_depth + 1 - min_depth)}."
            )

        individuals = []

        for depth in range(min_depth, max_depth + 1):
            # first create `group_size` individuals using grow method
            self.grow_creator.init_depth = (min_depth, depth)
            grown_inds = self.grow_creator.create_individuals(
                group_size, higher_is_better
            )

            # then create `group_size` individuals using full method
            self.full_creator.init_depth = (min_depth, depth)
            full_inds = self.full_creator.create_individuals(
                group_size, higher_is_better
            )

            individuals.extend(grown_inds + full_inds)

        # might need to add a few because 'group_size' may have been a float that was truncated
        n_missing = n_individuals - len(individuals)
        if n_missing > 0:
            missing_inds = self.full_creator.create_individuals(
                n_missing, higher_is_better
            )
            individuals.extend(missing_inds)

        self.created_individuals = individuals
        return individuals

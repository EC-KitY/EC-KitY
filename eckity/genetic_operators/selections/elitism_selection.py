from typing import List

from eckity.genetic_operators import SelectionMethod
from eckity.individual import Individual


class ElitismSelection(SelectionMethod):
    def __init__(self, num_elites, events=None):
        super().__init__(events=events)
        self.num_elites = num_elites

    def select(
        self, source_inds: List[Individual], dest_inds: List[Individual]
    ) -> List[Individual]:
        # assumes higher_is_better the same for all individuals
        higher_is_better = source_inds[0].higher_is_better

        elites = sorted(
            source_inds,
            key=lambda ind: ind.get_augmented_fitness(),
            reverse=higher_is_better,
        )[: self.num_elites]
        for elite in elites:
            cloned = elite.clone()
            cloned.selected_by.append(type(self).__name__)
            dest_inds.append(cloned)
        self.selected_individuals = dest_inds
        return dest_inds

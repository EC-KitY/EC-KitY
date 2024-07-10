from eckity.genetic_operators import SelectionMethod


class ElitismSelection(SelectionMethod):
    def __init__(self, num_elites, higher_is_better=False, events=None):
        super().__init__(events=events, higher_is_better=higher_is_better)
        self.num_elites = num_elites
        self.higher_is_better = higher_is_better

    def select(self, source_inds, dest_inds):
        elites = sorted(
            source_inds,
            key=lambda ind: ind.get_augmented_fitness(),
            reverse=self.higher_is_better,
        )[: self.num_elites]
        for elite in elites:
            cloned = elite.clone()
            cloned.selected_by.append(type(self).__name__)
            dest_inds.append(cloned)
        self.selected_individuals = dest_inds
        return dest_inds

from eckity.genetic_operators.selections.selection_method import SelectionMethod


class ElitismSelection(SelectionMethod):
    def __init__(self, num_elites, higher_is_better=False, events=None):
        super().__init__(events=events, higher_is_better=higher_is_better)
        self.num_elites = num_elites
        self.higher_is_better = higher_is_better

    def select(self, source_inds, dest_inds):
        elites = sorted(source_inds,
                        key=lambda ind: ind.get_augmented_fitness(),
                        reverse=self.higher_is_better)[:self.num_elites]
        for elite in elites:
            dest_inds.append(elite.clone())
        self.selected_individuals = dest_inds
        # TODO shouldn't it be after_operator? why is this needed?
        # self.publish("after_selection")
        return dest_inds

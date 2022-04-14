from random import choices

from eckity.genetic_operators.selections.selection_method import SelectionMethod
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection


class ProbabilisticMultiSelection(SelectionMethod):
    def __init__(self, probabilities=None, selection_methods=None, events=None):
        if events is None:
            events = ["after_choosing_selection_method"]
        super().__init__(events=events)
        if selection_methods is None:
            selection_methods = [TournamentSelection(1)]
        if probabilities is None:
            probabilities = [1]
        self.selection_methods = selection_methods
        self.probabilities = probabilities
        self.chosen_method = None

    def select(self, source_inds, dest_inds):
        self.chosen_method = choices(self.selection_methods, self.probabilities)[0]
        self.publish("after_choosing_selection_method")
        return self.chosen_method.select(self, source_inds, dest_inds, seed)

    def event_name_to_data(self, event_name):
        if event_name == "after_choosing_selection_method":
            return {"chosen_method": self.chosen_method}
        else:
            return super().event_name_to_data(event_name)

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, ProbabilisticMultiSelection)

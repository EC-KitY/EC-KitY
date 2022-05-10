from abc import abstractmethod

from eckity.event_based_operator import Operator


class SelectionMethod(Operator):
    def __init__(self, events=None, higher_is_better=False):
        if events is None:
            events = ["after_selection"]
        super().__init__(events=events)
        self.higher_is_better = higher_is_better
        self.selected_individuals = None

    def apply_operator(self, payload):
        return self.select(payload[0], payload[1])

    @abstractmethod
    def select(self, source_inds, dest_inds):
        pass

    def event_name_to_data(self, event_name):
        if event_name == "after_selection":
            return {"applied_individuals": self.selected_individuals}
        else:
            return {}

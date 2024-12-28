from abc import abstractmethod
from overrides import override

from typing import List

from eckity.event_based_operator import Operator
from eckity.individual import Individual


class SelectionMethod(Operator):
    def __init__(
        self,
        events=None,
    ):
        if events is None:
            events = ["after_selection"]
        super().__init__(events=events)
        self.selected_individuals = None

    @override
    def apply_operator(self, payload):
        return self.select(payload[0], payload[1])

    @abstractmethod
    def select(
        self, source_inds: List[Individual], dest_inds: List[Individual]
    ) -> List[Individual]:
        pass

    def event_name_to_data(self, event_name):
        if event_name == "after_selection":
            return {"applied_individuals": self.selected_individuals}
        else:
            return {}

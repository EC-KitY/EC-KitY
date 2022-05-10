from abc import abstractmethod

from eckity.event_based_operator import Operator


class Creator(Operator):
    def __init__(self, events=None):
        super().__init__(events)
        self.created_individuals = None

    @abstractmethod
    def create_individuals(self, n_individuals, higher_is_better):
        pass

    def apply_operator(self, payload):
        return self.create_individuals(payload)

    def event_name_to_data(self, event_name):
        if event_name == "after_operator":
            return {"created_individuals": self.created_individuals}
        else:
            return {}

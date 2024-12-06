from abc import abstractmethod
from typing import Any, Dict, List

from eckity.individual import Individual
from eckity.event_based_operator import Operator
from eckity.fitness.simple_fitness import SimpleFitness


class Creator(Operator):
    def __init__(self, events=None, fitness_type=SimpleFitness):
        super().__init__(events)
        self.created_individuals = None
        self.fitness_type = fitness_type

    @abstractmethod
    def create_individuals(
        self, n_individuals: int, higher_is_better: bool
    ) -> List[Individual]:
        pass

    def apply_operator(self, payload: Any) -> Any:
        return self.create_individuals(payload)

    def event_name_to_data(self, event_name: str) -> Dict[str, Any]:
        return (
            {"created_individuals": self.created_individuals}
            if event_name == "after_operator"
            else {}
        )

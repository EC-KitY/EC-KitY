from abc import abstractmethod

from overrides import overrides

from eckity.event_based_operator import Operator
from eckity.population import Population


class Breeder(Operator):
    """
    The Breeder is responsible to activate the genetic operators (selection, crossover, mutation)
    on the existing population

    Parameters
    ----------
    events: dict(str, dict(object, function))
        dictionary of event names to dictionary of subscribers to callback methods
    """
    def __init__(self,
                 events=None):
        super().__init__(events=events)

    def breed(self, population):
        """
        Breed the given population of the experiment.
        Hence, apply genetic operators on the individuals of the population.

        Parameters
        ----------
        population: Population
        The population of individuals existing in the current experiment.
        """
        self.act(population)

    @abstractmethod
    def apply_breed(self, population):
        pass

    @overrides
    def apply_operator(self, payload):
        population: Population = payload
        self.apply_breed(population)

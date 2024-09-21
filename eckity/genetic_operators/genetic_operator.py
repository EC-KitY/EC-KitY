import random
from abc import ABC, abstractmethod

from eckity.event_based_operator import Operator


class GeneticOperator(Operator, ABC):
    """
    Abstract class for genetic operators.
    Each operator has a probability of being applied each generation.

    Genetic operators are applied in-place.


    Parameters
    ----------
    probability : float, optional
        probability of being applied each generation, by default 1.0
    arity : int, optional
        number of individuals required for the operator, by default 0
    events : List[str], optional
        custom events that the operator publishes, by default None
    """

    def __init__(self, probability=1.0, arity=0, events=None):
        super().__init__(events=events, arity=arity)
        self.probability = probability

    def apply_operator(self, individuals):
        """
        Apply the genetic operator with a certain probability.
        The individuals are modified in-place, so it is not mandatory
        to return them.

        Parameters
        ----------
        individuals : List[Individual]
            Individuals to apply the operator to.

        Returns
        -------
        List[Individual]
            The individuals after applying the operator.
        """
        if random.random() <= self.probability:
            # Fitness is irrelevant once the operator is applied
            for individual in individuals:
                individual.set_fitness_not_evaluated()
            op_res = self.apply(individuals)

            # Add the operator to the applied operators list
            for ind in op_res:
                ind.applied_operators.append(type(self).__name__)
                
                if ind.update_parents:
                    parents = [p.id for p in individuals]
                    ind.parents.extend(parents)
            return op_res
        return individuals

    @abstractmethod
    def apply(self, individuals):
        """
        Apply the genetic operator to the individuals.
        This method should be implemented by the subclasses.

        Parameters
        ----------
        individuals : List[Individual]
            Individuals to apply the operator to.
        """
        pass

from random import choice, gauss

from eckity.event_based_operator import Operator
from eckity.genetic_operators.probabilistic_condition_operator import ProbabilisticConditionOperator

from eckity.genetic_operators.mutations.vector_one_point_mutation import VectorOnePointMutation


class VectorGaussOnePointFloatMutation(VectorOnePointMutation):
    def __init__(self, probability=1, arity=1, mu=0, sigma=1, events=None):
        mutated_value_getter = lambda individual, index: gauss(mu=mu, sigma=sigma)
        super().__init__(probability=probability, arity=arity, mutated_value_getter=mutated_value_getter, events=events)

    """def apply(self, individuals):
        
        #Perform ephemeral random constant (erc) mutation: select an erc node at random
        #and add Gaussian noise to it.

        #Returns
        #-------
        #None.
        
        for j in range(len(individuals)):
            m_point = choice(individuals[j].get_vector())
            value = round(individuals[j].get_cell_value(m_point) + gauss(mu=0, sigma=1))
            if value < individuals[j].get_bounds()[0] or value > individuals[j].get_bounds()[1]:
                value = get_random_number_in_bounds(individuals[j], m_point)
            # if out of bounds add val to normal random in range
            individuals[j].set_cell_value(m_point, value)
        self.applied_individuals = individuals
        return individuals"""


class IntVectorOnePointMutation(VectorOnePointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        mutated_value_getter = lambda individual, index: individual.get_random_number_in_bounds(index)
        super().__init__(probability=probability, arity=arity, mutated_value_getter=mutated_value_getter, events=events, n=1)


class IntVectorNPointMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None, n=1):
        mutated_value_getter = lambda individual, index: individual.get_random_number_in_bounds(index)
        super().__init__(probability=probability, arity=arity, mutated_value_getter=mutated_value_getter, events=events,
                         n=n)


class BitStringVectorFlipMutation(VectorOnePointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        mutated_value_getter = lambda individual, index: individual.get_random_number_in_bounds(index)
        super().__init__(probability=probability, arity=arity, mutated_value_getter=mutated_value_getter, events=events, n=1)

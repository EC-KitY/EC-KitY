from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation

#from eckity.genetic_operators.mutations.vector_one_point_mutation import VectorOnePointMutation, VectorNPointMutation



class VectorGaussOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, mu=0, sigma=1, events=None):
        mutated_value_getter = lambda individual, index: individual.gauss(mu=mu, sigma=sigma)
        super().__init__(probability=probability, arity=arity, mut_val_getter=mutated_value_getter, events=events,
                         n=1)

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
class VectorNPointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index, sigma),
                         events=events)


class VectorGaussNPointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, sigma),
                         events=events)


class IntVectorOnePointMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        mutated_value_getter = lambda individual, index: individual.get_random_number_in_bounds(index)
        super().__init__(probability=probability, arity=arity, mut_val_getter=mutated_value_getter, events=events,
                         n=1)


class IntVectorNPointMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None, n=1):
        mutated_value_getter = lambda individual, index: individual.get_random_number_in_bounds(index)
        super().__init__(probability=probability, arity=arity, mut_val_getter=mutated_value_getter, events=events,
                         n=n)


class BitStringVectorFlipMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events, n=1)

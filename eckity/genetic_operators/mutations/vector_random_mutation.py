from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation


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
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events)

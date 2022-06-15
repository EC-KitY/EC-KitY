from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation

# TODO add mu argument to gauss methods

class VectorUniformNPointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index, sigma),
                         events=events,
                         on_fail=lambda inds: True)


class VectorGaussNPointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, sigma),
                         events=events,
                         # TODO on fail = lambda that invokes uniform mutation for each individual
                         )


# TODO multiple classes?
class VectorUniformOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index, sigma),
                         events=events,
                         on_fail=lambda inds: True)


class VectorGaussOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, sigma),
                         events=events,
                         # TODO on fail = lambda that invokes uniform mutation for each individual
                         )


class IntVectorOnePointMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         on_fail=lambda inds: True)

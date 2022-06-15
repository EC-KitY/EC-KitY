from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation


class VectorUniformOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events,
                         on_fail=lambda individuals: None)


class VectorUniformNPointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1, arity=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events,
                         on_fail=lambda individuals: None)


class VectorGaussNPointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, mu=0, sigma=1, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events,
                         # TODO on fail = lambda that invokes uniform mutation for each individual
                         )


class VectorGaussOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1, arity=1, sigma=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events,
                         # TODO on fail = lambda that invokes uniform mutation for each individual
                         )


class IntVectorOnePointMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         n=1,
                         on_fail=lambda individuals: None)


class IntVectorNPointMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None, n=1):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         n=n,
                         on_fail=lambda individuals: None)


class BitStringVectorFlipMutation(VectorNPointMutation):
    def __init__(self, probability=1, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         n=1,
                         events=events,
                         on_fail=lambda individuals: True)

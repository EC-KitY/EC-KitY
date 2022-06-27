from random import random

from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation


class VectorUniformOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events,
                         on_fail=lambda individuals: None)


class VectorUniformNPointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1.0, arity=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events,
                         on_fail=lambda individuals: None)


class VectorGaussOnePointFloatMutation(VectorNPointMutation):
    def __init__(self, probability=1.0, arity=1, mu=0.0, sigma=1.0, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events,
                         on_fail=on_gauss_fail(1, probability, arity, events))


class VectorGaussNPointFloatMutation(VectorNPointMutation):
    def __init__(self, n=1, probability=1.0, arity=1, mu=0.0, sigma=1.0, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events,
                         on_fail=on_gauss_fail(n, probability, arity, events))


def on_gauss_fail(n=1, probability=1.0, arity=1, events=None):
    """
    Handle gauss mutation failure by returning a callable uniform mutation
    """
    mut = VectorUniformNPointFloatMutation(n, probability, arity, events)
    return mut.apply_operator


class IntVectorOnePointMutation(VectorNPointMutation):
    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         n=1,
                         on_fail=lambda individuals: None)


class IntVectorNPointMutation(VectorNPointMutation):
    def __init__(self, probability=1.0, arity=1, events=None, n=1):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         n=n,
                         on_fail=lambda individuals: None)


class BitStringVectorFlipMutation(VectorNPointMutation):
    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.flip(index),
                         n=1,
                         events=events,
                         on_fail=lambda individuals: True)


class BitStringVectorNFlipMutation(VectorNPointMutation):
    def __init__(self, probability=1.0, arity=1, events=None, probability_for_each=0.2,n=100):
        self.probability_for_each = probability_for_each
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.flip(
                             index) if random() <= self.probability_for_each else individual.cell_value(index),
                         events=events,
                         on_fail=lambda individuals: True,
                         n=n)

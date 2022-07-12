from random import random

from eckity.genetic_operators.mutations.vector_n_point_mutation import VectorNPointMutation


class VectorUniformOnePointFloatMutation(VectorNPointMutation):
    """
    Uniform One Point Float Mutation
    """
    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events)


class VectorUniformNPointFloatMutation(VectorNPointMutation):
    """
    Uniform N Point Float Mutation
    """
    def __init__(self, n=1, probability=1.0, arity=1, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(index),
                         events=events)


class VectorGaussOnePointFloatMutation(VectorNPointMutation):
    """
    Gaussian One Point Float Mutation
    """
    def __init__(self, probability=1.0, arity=1, mu=0.0, sigma=1.0, events=None):
        super().__init__(n=1,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events)

    def on_fail(self, payload):
        """
        Handle gauss mutation failure by returning a callable uniform mutation
        """
        mut = VectorUniformNPointFloatMutation(1, self.probability, self.arity, self.events)
        return mut.apply_operator


class VectorGaussNPointFloatMutation(VectorNPointMutation):
    """
    Gaussian N Point Float Mutation
    """
    def __init__(self, n=1, probability=1.0, arity=1, mu=0.0, sigma=1.0, events=None):
        super().__init__(n=n,
                         probability=probability,
                         arity=arity,
                         mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(index, mu, sigma),
                         events=events)

    def on_fail(self, payload):
        """
        Handle gauss mutation failure by returning a callable uniform mutation
        """
        mut = VectorUniformNPointFloatMutation(self.n, self.probability, self.arity, self.events)
        return mut.apply_operator


class IntVectorOnePointMutation(VectorNPointMutation):
    """
    Uniform One Point Integer Mutation
    """
    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         n=1)


class IntVectorNPointMutation(VectorNPointMutation):
    """
    Uniform N Point Integer Mutation
    """
    def __init__(self, probability=1.0, arity=1, events=None, n=1):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(index),
                         events=events,
                         n=n)


class BitStringVectorFlipMutation(VectorNPointMutation):
    """
    One Point Bit-Flip Mutation
    """
    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.bit_flip(index),
                         n=1,
                         events=events)


class BitStringVectorNFlipMutation(VectorNPointMutation):
    """
    N Point Bit-Flip Mutation
    """
    def __init__(self, probability=1.0, arity=1, events=None, probability_for_each=0.2, n=100):
        self.probability_for_each = probability_for_each
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.bit_flip(
                             index) if random() <= self.probability_for_each else individual.cell_value(index),
                         events=events,
                         n=n)

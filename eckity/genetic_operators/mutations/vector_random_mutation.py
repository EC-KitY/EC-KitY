from random import random

from eckity.genetic_operators.mutations.vector_n_point_mutation import (
    VectorNPointMutation,
)

"""
This module contains the implementation of various vector mutations
The key difference between these mutations is the way they generate the new
value for the mutated cell (mut_val_getter).

Additional parameters are used to control the mutation process, such as
the way of selecting cells to mutate (cell_selector),
the way of checking if the new value is legal (success_checker).
For additional information, see the VectorNPointMutation class.

Standard parameters for the mutation are number of mutation points (n),
the probability of the operator itself to occur (probability),
the probability of each point to be mutated (probability_for_each),
the number of attempts to be made, etc.
"""


class FloatVectorUniformOnePointMutation(VectorNPointMutation):
    """
    Uniform One Point Float Mutation.
    Mutates a single cell of a float vector.
    Mutated value is drawn from a uniform distribution,
    with respect to the bounds of the vector.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    events : List[str], optional
        custom events to be published by the mutation, by default None
    """

    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(
            n=1,
            probability=probability,
            arity=arity,
            mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(
                index
            ),
            events=events,
        )


class FloatVectorUniformNPointMutation(VectorNPointMutation):
    """
    Uniform N Point Float Mutation.
    Mutates exactly n cells of a float vector.
    Mutated values are drawn from a uniform distribution,
    with respect to the bounds of the vector.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    events : List[str], optional
        custom events to be published by the mutation, by default None
    """

    def __init__(self, n=1, probability=1.0, arity=1, events=None):
        super().__init__(
            n=n,
            probability=probability,
            arity=arity,
            mut_val_getter=lambda vec, index: vec.get_random_number_in_bounds(
                index
            ),
            events=events,
        )


class FloatVectorGaussOnePointMutation(VectorNPointMutation):
    """
    Gaussian One Point Float Mutation.
    Mutates a single cell of a float vector.
    Mutated value is drawn from a Gaussian
    distribution with mean mu and standard deviation sigma.

    The mutation is repeated until the new value is legal
    (within the bounds of the vector), or the number of attempts
    is exceeded.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    mu : float, optional
        Gaussian distribution mean value, by default 0.0
    sigma : float, optional
        Gaussian distribution std value, by default 1.0
    events : List[str], optional
        custom events to be published by the mutation, by default None
    attempts : int, optional
        number of attempts till failure, by default 5
    """

    def __init__(
            self,
            probability=1.0,
            arity=1,
            mu=0.0,
            sigma=1.0,
            events=None,
            attempts=5,
    ):
        super().__init__(
            n=1,
            probability=probability,
            arity=arity,
            mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(
                index, mu, sigma
            ),
            events=events,
            attempts=attempts,
        )

    def on_fail(self, payload):
        """
        Handle gauss mutation failure by performing uniform mutation.
        """
        mut = FloatVectorUniformOnePointMutation(
            self.probability, self.arity, self.events
        )
        return mut.apply_operator(payload)


class FloatVectorGaussNPointMutation(VectorNPointMutation):
    """
    Gaussian N Point Float Mutation.
    Mutates exactly n cells of a float vector.
    Mutated values are drawn from a Gaussian
    distribution with mean mu and standard deviation sigma.

    The mutation is repeated until the new value is legal
    (within the bounds of the vector), or the number of attempts
    is exceeded.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    mu : float, optional
        Gaussian distribution mean value, by default 0.0
    sigma : float, optional
        Gaussian distribution std value, by default 1.0
    events : List[str], optional
        custom events to be published by the mutation, by default None
    attempts : int, optional
        number of attempts till failure, by default 5
    """

    def __init__(
            self,
            n=1,
            probability=1.0,
            arity=1,
            mu=0.0,
            sigma=1.0,
            events=None,
            attempts=5,
    ):
        super().__init__(
            n=n,
            probability=probability,
            arity=arity,
            mut_val_getter=lambda vec, index: vec.get_random_number_with_gauss(
                index, mu, sigma
            ),
            events=events,
            attempts=attempts,
        )

    def on_fail(self, payload):
        """
        Handle gauss mutation failure by performing uniform mutation.
        """
        mut = FloatVectorUniformNPointMutation(
            self.n, self.probability, self.arity, self.events
        )
        return mut.apply_operator(payload)


class IntVectorOnePointMutation(VectorNPointMutation):
    """
    Uniform one point Int mutation.
    Mutates a single cell of a int vector.
    Mutated value is drawn from a uniform distribution,
    with respect to the bounds of the vector.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    events : List[str], optional
        custom events to be published by the mutation, by default None
    """

    def __init__(self, probability=0.5, arity=1, events=None, probability_for_each=0.1):
        self.probability_for_each = probability_for_each
        super().__init__(probability=probability,
                         arity=arity,
                         mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(
                             index) if random() <= self.probability_for_each else individual.cell_value(index),
                         events=events, cell_selector=lambda vec: list(range(vec.size())))


class IntVectorNPointMutation(VectorNPointMutation):
    """
    Uniform N point Int mutation.
    Mutates exactly n cells of a int vector.
    Mutated value is drawn from a uniform distribution,
    with respect to the bounds of the vector.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    events : List[str], optional
        custom events to be published by the mutation, by default None
    """

    def __init__(self, probability=1.0, arity=1, events=None, n=1):
        super().__init__(
            probability=probability,
            arity=arity,
            mut_val_getter=lambda individual, index: individual.get_random_number_in_bounds(
                index
            ),
            events=events,
            n=n,
        )


class BitStringVectorFlipMutation(VectorNPointMutation):
    """
    One Point Bit-Flip Mutation
    Flips a single bit of a bit vector.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    events : List[str], optional
        custom events to be published by the mutation, by default None
    """

    def __init__(self, probability=1.0, arity=1, events=None):
        super().__init__(
            probability=probability,
            arity=arity,
            mut_val_getter=lambda individual, index: individual.bit_flip(
                index
            ),
            n=1,
            events=events,
        )


class BitStringVectorNFlipMutation(VectorNPointMutation):
    """
    Multiple Bit-Flip Mutation
    Traverses the bit vector and flips each bit with a certain probability
    (probability_for_each).
    Note that this mutation is not guaranteed to flip an exact number of bits,
    as the flipping is done with a certain probability.

    Parameters
    ----------
    probability : float, optional
        probability of the operator to occur, by default 1.0
    arity : int, optional
        individuals required for the mutation, by default 1
    events : List[str], optional
        custom events to be published by the mutation, by default None
    probability_for_each=0.2 : float, optional
        probability of flipping each bit, by default 0.2
    """

    def __init__(
            self,
            probability=1.0,
            arity=1,
            events=None,
            probability_for_each=0.2,
            n=1,
    ):
        self.probability_for_each = probability_for_each
        super().__init__(
            probability=probability,
            arity=arity,
            mut_val_getter=lambda individual, index: (
                individual.bit_flip(index)
                if random() <= self.probability_for_each
                else individual.cell_value(index)
            ),
            events=events,
            n=n
        )

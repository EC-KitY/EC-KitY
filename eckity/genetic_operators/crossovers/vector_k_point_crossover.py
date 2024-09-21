from random import sample

from eckity.genetic_operators.genetic_operator import GeneticOperator
from eckity.genetic_encodings.ga import Vector

from typing import List, Tuple


class VectorKPointsCrossover(GeneticOperator):
    def __init__(self, probability=1, arity=2, k=1, events=None):
        """
        Vector K Point Crossover.

        Randomly chooses K points in the vector and swaps the parts
        of two vectors at these points.

        Parameters
        ----------
        probability : float
            The probability of the mutation operator to be applied

        arity : int
            The number of individuals this mutation is applied on

        k : int
            Number of points to cut the vector for the crossover.

        events: list of strings
            Custom events to be published by the mutation, by default None
        """
        self.individuals = None
        self.applied_individuals = None
        self.k = k
        super().__init__(probability=probability, arity=arity, events=events)

    def apply(self, individuals: List[Vector]) -> List[Vector]:
        """
        Attempt to perform the mutation operator

        Parameters
        ----------
        individuals : List[Vector]
            individuals to perform crossover on

        Returns
        ----------
        List[Vector]
            individuals after the crossover
        """
        self.individuals = individuals
        xo_points = sorted(sample(range(1, individuals[0].size()), self.k))
        self._swap_vector_parts(
            individuals[0].vector, individuals[1].vector, xo_points
        )

        self.applied_individuals = individuals
        return individuals

    def _swap_vector_parts(
        self, vector1: List[int], vector2: List[int], xo_points: List[int]
    ) -> Tuple[List[int]]:
        """
        Swap parts of two vectors at the given crossover points.

        Parameters
        ----------
        vector1 : List[int]
            first vector encoding
        vector2 : List[int]
            second vector encoding
        xo_points : List[int]
            crossover points

        Returns
        -------
        Tuple[List[int]]
            _description_
        """
        if len(xo_points) == self.k:
            xo_points.append(len(vector1))

        start_idx = 0
        for i in range(0, len(xo_points), 2):
            end_idx = xo_points[i]
            replaced_part = vector1[start_idx:end_idx]
            vector1[start_idx:end_idx] = vector2[start_idx:end_idx]
            vector2[start_idx:end_idx] = replaced_part
            start_idx = xo_points[i + 1] if end_idx < len(xo_points) else -1

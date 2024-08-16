from eckity.genetic_encodings.ga.int_vector import IntVector
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.fitness.simple_fitness import SimpleFitness


class TestVectorKPointCrossover:
    def test_one_point_crossover(self):
        v1 = [1, 2, 3, 4]
        v2 = [5, 6, 7, 8]
        xo_points = [2]

        crossover = VectorKPointsCrossover(k=1)
        crossover._swap_vector_parts(v1, v2, xo_points)

        assert v1 == [5, 6, 3, 4]
        assert v2 == [1, 2, 7, 8]

    def test_two_point_crossover(self):
        v1 = [1, 2, 3, 4]
        v2 = [5, 6, 7, 8]
        xo_points = [1, 3]

        crossover = VectorKPointsCrossover(k=2)
        crossover._swap_vector_parts(v1, v2, xo_points)

        assert v1 == [5, 2, 3, 8]
        assert v2 == [1, 6, 7, 4]

    def test_applied_operators(self):
        length = 4
        v1 = IntVector(
            SimpleFitness(), length, bounds=(1, 10), vector=[1, 2, 3, 4]
        )
        v2 = IntVector(
            SimpleFitness(), length, bounds=(1, 10), vector=[5, 6, 7, 8]
        )

        crossover = VectorKPointsCrossover(k=1)
        crossover.apply_operator([v1, v2])
        assert v1.applied_operators == ["VectorKPointsCrossover"]
        assert v2.applied_operators == ["VectorKPointsCrossover"]
import random
from eckity.genetic_encodings.ga.int_vector import IntVector
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.fitness.simple_fitness import SimpleFitness


def test_vector_two_point_crossover():
    random.seed(0)
    length = 4
    v1 = IntVector(SimpleFitness(), length, bounds=(1, 10))
    v1.set_vector(list(range(1, 5)))
    v2 = IntVector(SimpleFitness(), length, bounds=(1, 10))
    v2.set_vector(list(range(5, 9)))

    # random sample will return [2, 3]
    expected_v1 = [5, 2, 7, 8]
    expected_v2 = [1, 6, 3, 4]

    crossover = VectorKPointsCrossover(k=2)
    crossover.apply_operator([v1, v2])
    assert v1.vector == expected_v1
    assert v2.vector == expected_v2
    assert v1.applied_operators == ["VectorKPointsCrossover"]
    assert v2.applied_operators == ["VectorKPointsCrossover"]

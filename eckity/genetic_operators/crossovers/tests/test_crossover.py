import random
from eckity.genetic_encodings.ga.int_vector import IntVector
from eckity.genetic_operators.crossovers.vector_k_point_crossover \
    import VectorKPointsCrossover
from eckity.fitness.simple_fitness import SimpleFitness


class TestCrossover:
    def test_vector_two_point_crossover(self):
        random.seed(0)
        length = 4

        og_v1 = list(range(1, 5))
        og_v2 = list(range(5, 9))

        v1 = IntVector(SimpleFitness(),
                       length,
                       bounds=(1, 10),
                       update_parents=True)
        v1.set_vector(og_v1)
        v2 = IntVector(SimpleFitness(),
                       length,
                       bounds=(1, 10),
                       update_parents=True)
        v2.set_vector(og_v2)

        # random sample will return [2, 3]
        expected_v1 = [5, 2, 7, 8]
        expected_v2 = [1, 6, 3, 4]

        crossover = VectorKPointsCrossover(k=2)
        crossover.apply_operator([v1, v2])
        assert v1.vector == expected_v1
        assert v2.vector == expected_v2
        assert v1.applied_operators == ['VectorKPointsCrossover']
        assert v2.applied_operators == ['VectorKPointsCrossover']

        # test parents field
        for v in [v1, v2]:
            assert len(v.parents) == 2
            assert v.parents[0].get_vector() == og_v1
            assert v.parents[1].get_vector() == og_v2
        

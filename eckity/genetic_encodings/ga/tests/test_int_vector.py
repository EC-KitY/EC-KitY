import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.int_vector import IntVector


class TestIntVector:
    def test_vector_direct_initialization(self):
        length = 5
        cells = [0, 1, 1, 1, 3]
        vec = IntVector(SimpleFitness(), length=length, vector=cells)
        assert len(vec.vector) == length
        assert vec.vector == cells

    def test_bad_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]

        with pytest.raises(ValueError):
            IntVector(SimpleFitness(), bounds=bounds, length=length + 1)
        with pytest.raises(ValueError):
            IntVector(SimpleFitness(), bounds=bounds, length=length - 1)

    def test_int_get_rand_num_single_bounds(self):
        bounds = (1, 5)
        vec1 = IntVector(SimpleFitness(), bounds=bounds, length=5)

        for _ in range(10**3):
            num = vec1.get_random_number_in_bounds(0)
            assert isinstance(num, int) and bounds[0] <= num <= bounds[1]

    def test_int_get_rand_num_multi_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]
        vec1 = IntVector(SimpleFitness(), bounds=bounds, length=length)

        for i in range(length):
            num = vec1.get_random_number_in_bounds(i)
            assert isinstance(num, int) and bounds[i][0] <= num <= bounds[i][1]

    def test_get_vector_part_last_cell(self):
        length = 4
        v1 = IntVector(SimpleFitness(), length, bounds=(1, 10))
        v1.set_vector(list(range(1, 5)))
        assert v1.get_vector_part(length - 1, length) == [4]

    def test_clone(self):
        cells = [0, 1]
        score = 0.1
        v1 = IntVector(SimpleFitness(score, cache=True),
                       length=len(cells),
                       vector=cells)

        v2 = v1.clone()
        assert v2.vector == v1.vector
        assert v2.length == v1.length
        assert v2.bounds == v1.bounds
        assert v2.get_pure_fitness() == v1.get_pure_fitness()
        assert v1.id == v2.id - 1
        assert v2.cloned_from == [v1.id]

        # Check that fitness is evaluated and equal to original one
        assert v2.fitness.get_pure_fitness() == score
        assert v2.fitness.is_fitness_evaluated()

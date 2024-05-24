import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.float_vector import FloatVector


class TestFloatVector:
    def test_vector_direct_initialization(self):
        length = 5
        cells = [0.0, 0.0, 1.0, 1.0, 1.0]
        vec = FloatVector(SimpleFitness(), length=length, vector=cells)
        assert len(vec.vector) == length
        assert vec.vector == cells

    def test_bad_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]

        with pytest.raises(ValueError):
            FloatVector(SimpleFitness(), bounds=bounds, length=length + 1)
        with pytest.raises(ValueError):
            FloatVector(SimpleFitness(), bounds=bounds, length=length - 1)

    def test_get_vector_part_last_cell(self):
        length = 4
        vec = FloatVector(SimpleFitness(), length, bounds=(1, 10))

        vec.set_vector(list(range(1, 5)))
        assert vec.get_vector_part(length - 1, length) == [4]

    def test_clone(self):
        cells = [1.0, 2.0]
        score = 0.1
        v1 = FloatVector(SimpleFitness(score, cache=True),
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

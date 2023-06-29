import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.float_vector import FloatVector


class TestFloatVector:
	def test_vector_direct_initialization(self):
		length = 5
		cells = [0., 0., 1., 1., 1.]
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

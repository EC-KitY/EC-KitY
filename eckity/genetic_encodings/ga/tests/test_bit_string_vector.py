import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class TestBitStringVector:
    def test_vector_direct_initialization(self):
        length = 5
        cells = [0, 0, 1, 1, 1]
        vec = BitStringVector(SimpleFitness(),
                              length=length,
                              vector=cells)
        assert len(vec.vector) == length
        assert vec.vector == cells

    def test_bad_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]

        with pytest.raises(ValueError):
            BitStringVector(SimpleFitness(), bounds=bounds, length=length + 1)
        with pytest.raises(ValueError):
            BitStringVector(SimpleFitness(), bounds=bounds, length=length - 1)

    def test_replace_vector_part_bit_index0(self):
        vec_len = 5
        vec1 = BitStringVector(SimpleFitness(), length=vec_len)
        vec1.vector = [0] * vec_len

        vec2 = BitStringVector(SimpleFitness(), length=vec_len)
        vec2.vector = [1] * vec_len

        replaced_part_sz = 3
        old_vec1_part = vec1.replace_vector_part(
            vec2.vector[:replaced_part_sz], 0
        )

        expected1 = [1] * replaced_part_sz + [0] * (vec_len - replaced_part_sz)
        assert vec1.vector == expected1

        old_vec2_part = vec2.replace_vector_part(old_vec1_part, 0)
        assert old_vec2_part == [1] * replaced_part_sz

        expected2 = [0] * replaced_part_sz + [1] * (vec_len - replaced_part_sz)
        assert vec2.vector == expected2

    def test_replace_vector_part_bit_middle_index(self):
        vec1 = BitStringVector(SimpleFitness(), length=4)
        vec1.vector = [0, 1] * 2

        vec2 = BitStringVector(SimpleFitness(), length=4)
        vec2.vector = [1, 0] * 2

        old_vec1_part = vec1.replace_vector_part(vec2.vector[1:3], 1)
        assert vec1.vector == [0, 0, 1, 1]
        assert old_vec1_part == [1, 0]

        old_vec2_part = vec2.replace_vector_part(old_vec1_part, 1)
        assert old_vec2_part == [0, 1]
        assert vec2.vector == [1, 1, 0, 0]

    def test_bit_flip(self):
        vec1 = BitStringVector(SimpleFitness(), length=4)
        init_vec = [0, 1, 0, 1]
        vec1.vector = init_vec.copy()

        assert vec1.bit_flip(0) == vec1.bit_flip(2) == 1
        assert vec1.bit_flip(1) == vec1.bit_flip(3) == 0
        assert init_vec == vec1.vector

    def test_bit_get_rand_num_single_bounds(self):
        bounds = (0, 1)
        vec1 = BitStringVector(SimpleFitness(), bounds=bounds, length=5)

        for _ in range(10 ** 3):
            num = vec1.get_random_number_in_bounds(0)
            assert isinstance(num, int) and num == bounds[0] or num == bounds[1]

    def test_get_vector_part_last_cell(self):
        length = 2
        v1 = BitStringVector(SimpleFitness(), length, vector=[0, 1])
        assert v1.get_vector_part(length - 1, length) == [1]

    def test_clone(self):
        cells = [0, 1]
        score = 0.1
        v1 = BitStringVector(SimpleFitness(score, cache=True),
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

    def test_parents(self):
        v1 = BitStringVector(SimpleFitness(), length=2, update_parents=True)
        v2 = BitStringVector(SimpleFitness(), length=2, update_parents=True)

        v1.parents = [v2]
        assert v1.parents == [v2]

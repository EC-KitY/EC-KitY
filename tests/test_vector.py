from collections import Counter

import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector
from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.genetic_encodings.ga.int_vector import IntVector
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorGaussOnePointMutation, \
    FloatVectorUniformNPointMutation, FloatVectorGaussNPointMutation, IntVectorOnePointMutation, \
    BitStringVectorNFlipMutation, IntVectorNPointMutation


class TestVector:
    def test_bad_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]

        with pytest.raises(ValueError):
            FloatVector(SimpleFitness(), bounds=bounds, length=length + 1)
        with pytest.raises(ValueError):
            FloatVector(SimpleFitness(), bounds=bounds, length=length - 1)

    def test_replace_vector_part_bit_index0(self):
        vec_len = 5
        vec1 = BitStringVector(SimpleFitness(), length=vec_len)
        vec1.vector = [0] * vec_len

        vec2 = BitStringVector(SimpleFitness(), length=vec_len)
        vec2.vector = [1] * vec_len

        replaced_part_size = 3
        old_vec1_part = vec1.replace_vector_part(vec2.vector[:replaced_part_size], 0)
        assert vec1.vector == [1] * replaced_part_size + [0] * (vec_len - replaced_part_size)

        old_vec2_part = vec2.replace_vector_part(old_vec1_part, 0)
        assert old_vec2_part == [1] * replaced_part_size
        assert vec2.vector == [0] * replaced_part_size + [1] * (vec_len - replaced_part_size)

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

    def test_gauss_mutation_success(self):
        length = 4
        vec1 = FloatVector(SimpleFitness(), length=length, bounds=(-100.0, 100.0))
        init_vec = [2.0] * length
        vec1.vector = init_vec.copy()
        mut = FloatVectorGaussOnePointMutation()

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == 2
        assert cnt[2.0] == length - 1

    def test_gauss_mutation_fail(self):
        length = 4
        vec1 = FloatVector(SimpleFitness(), length=length, bounds=(-1.0, 1.0))
        init_vec = [1.0] * length
        vec1.vector = init_vec.copy()
        mut = FloatVectorGaussOnePointMutation(mu=1000)

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == 2
        assert cnt[1.0] == length - 1

    def test_float_get_rand_num_single_bounds(self):
        bounds = (-1.0, 1.0)
        vec1 = FloatVector(SimpleFitness(), bounds=bounds, length=5)

        for _ in range(10 ** 3):
            num = vec1.get_random_number_in_bounds(0)
            assert type(num) == float and bounds[0] <= num <= bounds[1]

    def test_int_get_rand_num_single_bounds(self):
        bounds = (1, 5)
        vec1 = IntVector(SimpleFitness(), bounds=bounds, length=5)

        for _ in range(10 ** 3):
            num = vec1.get_random_number_in_bounds(0)
            assert type(num) == int and bounds[0] <= num <= bounds[1]

    def test_bit_get_rand_num_single_bounds(self):
        bounds = (0, 1)
        vec1 = BitStringVector(SimpleFitness(), bounds=bounds, length=5)

        for _ in range(10 ** 3):
            num = vec1.get_random_number_in_bounds(0)
            assert type(num) == int and num == bounds[0] or num == bounds[1]

    def test_float_get_rand_num_multi_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]
        vec1 = FloatVector(SimpleFitness(), bounds=bounds, length=length)

        for i in range(length):
            num = vec1.get_random_number_in_bounds(i)
            assert type(num) == float and bounds[i][0] <= num <= bounds[i][1]

    def test_int_get_rand_num_multi_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]
        vec1 = IntVector(SimpleFitness(), bounds=bounds, length=length)

        for i in range(length):
            num = vec1.get_random_number_in_bounds(i)
            assert type(num) == int and bounds[i][0] <= num <= bounds[i][1]

    def test_uniform_bit_n_point_mut(self):
        length = 5
        n_points = 3
        vec1 = BitStringVector(SimpleFitness(), length=length)
        init_vec = [0] * length
        vec1.vector = init_vec.copy()
        mut = BitStringVectorNFlipMutation(n=n_points, probability_for_each=1.0)

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == 2
        assert cnt[0] == length - n_points
        assert cnt[1] == n_points

    def test_uniform_int_n_point_mut(self):
        length = 5
        n_points = 3
        vec1 = IntVector(SimpleFitness(), length=length, bounds=(-10000, 10000))
        init_vec = [0] * length
        vec1.vector = init_vec.copy()
        mut = IntVectorNPointMutation(n=n_points)

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == n_points + 1
        assert cnt[0] == length - n_points

    def test_uniform_float_n_point_mut(self):
        length = 5
        n_points = 3
        vec1 = FloatVector(SimpleFitness(), length=length, bounds=(-100.0, 100.0))
        init_vec = [0.0] * length
        vec1.vector = init_vec.copy()
        mut = FloatVectorUniformNPointMutation(n=n_points)

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == n_points + 1
        assert cnt[0.0] == length - n_points

    def test_gauss_float_n_point_mut(self):
        length = 5
        n_points = 3
        vec1 = FloatVector(SimpleFitness(), length=length, bounds=(-100.0, 100.0))
        init_vec = [0.0] * length
        vec1.vector = init_vec.copy()
        mut = FloatVectorGaussNPointMutation(n=n_points)

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == n_points + 1
        assert cnt[0.0] == length - n_points

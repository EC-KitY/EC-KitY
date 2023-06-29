from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.genetic_operators.mutations.vector_random_mutation \
    import FloatVectorUniformNPointMutation, \
    FloatVectorGaussNPointMutation, FloatVectorGaussOnePointMutation

from eckity.fitness.simple_fitness import SimpleFitness
from collections import Counter


class TestMutations:
    def test_uniform_float_n_point_mut(self):
        length = 5
        n_points = 3
        vec1 = FloatVector(SimpleFitness(),
                           length=length, bounds=(-100.0, 100.0))
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
        vec = FloatVector(SimpleFitness(),
                          length=length, bounds=(-100.0, 100.0))
        init_vec = [0.0] * length
        vec.vector = init_vec.copy()
        mut = FloatVectorGaussNPointMutation(n=n_points)

        mut.apply_operator([vec])
        cnt = Counter(vec.vector)

        assert vec.vector != init_vec
        assert len(cnt.keys()) == n_points + 1
        assert cnt[0.0] == length - n_points

    def test_gauss_mutation_success(self):
        length = 4
        vec = FloatVector(SimpleFitness(),
                          length=length, bounds=(-100.0, 100.0))
        init_vec = [2.0] * length
        vec.vector = init_vec.copy()
        mut = FloatVectorGaussOnePointMutation()

        mut.apply_operator([vec])
        cnt = Counter(vec.vector)

        assert vec.vector != init_vec
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

    def test_float_get_rand_num_multi_bounds(self):
        length = 5
        bounds = [(i, i + 1) for i in range(length)]
        vec = FloatVector(SimpleFitness(), bounds=bounds, length=length)

        for i in range(length):
            num = vec.get_random_number_in_bounds(i)
            assert type(num) == float and bounds[i][0] <= num <= bounds[i][1]

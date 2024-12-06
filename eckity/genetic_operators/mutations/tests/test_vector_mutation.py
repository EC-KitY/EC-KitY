from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.genetic_operators.mutations.vector_random_mutation import (
    FloatVectorUniformNPointMutation,
    FloatVectorGaussNPointMutation,
    FloatVectorGaussOnePointMutation,
)

from eckity.fitness.simple_fitness import SimpleFitness
from collections import Counter


class TestMutations:
    def test_uniform_float_n_point_mut(self):
        length = 5
        n_points = 3
        vec1 = FloatVector(
            SimpleFitness(0.0),
            length=length,
            bounds=(-100.0, 100.0),
            update_parents=True,
        )
        init_vec = [0.0] * length
        vec1.vector = init_vec.copy()
        mut = FloatVectorUniformNPointMutation(n=n_points, probability=1.0)

        mut.apply_operator([vec1])
        cnt = Counter(vec1.vector)

        assert len(cnt.keys()) == n_points + 1
        assert cnt[0.0] == length - n_points
        assert vec1.applied_operators == ["FloatVectorUniformNPointMutation"]

        assert len(vec1.parents) == 1
        assert vec1.id in vec1.parents

    def test_gauss_float_n_point_mut(self):
        length = 5
        n_points = 3
        vec = FloatVector(
            SimpleFitness(0.0), length=length, bounds=(-100.0, 100.0)
        )
        init_vec = [0.0] * length
        vec.vector = init_vec.copy()
        mut = FloatVectorGaussNPointMutation(n=n_points, probability=1.0)

        mut.apply_operator([vec])
        cnt = Counter(vec.vector)

        assert vec.vector != init_vec
        assert len(cnt.keys()) == n_points + 1
        assert cnt[0.0] == length - n_points
        assert vec.applied_operators == ["FloatVectorGaussNPointMutation"]

    def test_gauss_mutation_success(self):
        length = 4
        vec = FloatVector(
            SimpleFitness(0.0), length=length, bounds=(-100.0, 100.0)
        )
        init_vec = [2.0] * length
        vec.vector = init_vec.copy()
        mut = FloatVectorGaussOnePointMutation(probability=1.0)

        mut.apply_operator([vec])
        cnt = Counter(vec.vector)

        assert vec.vector != init_vec
        assert len(cnt.keys()) == 2
        assert cnt[2.0] == length - 1
        assert vec.applied_operators == ["FloatVectorGaussOnePointMutation"]

    def test_gauss_mutation_fail(self):
        length = 4
        vec = FloatVector(
            SimpleFitness(0.0), length=length, bounds=(-1.0, 1.0)
        )
        assert vec.fitness.is_fitness_evaluated()

        init_vec = [1.0] * length
        vec.vector = init_vec.copy()
        mut = FloatVectorGaussOnePointMutation(mu=1000, probability=1.0)

        mut.apply_operator([vec])

        cnt = Counter(vec.vector)

        assert len(cnt.keys()) == 2
        assert cnt[1.0] == length - 1
        assert vec.applied_operators == [
            "FloatVectorUniformOnePointMutation",
            "FloatVectorGaussOnePointMutation",
        ]

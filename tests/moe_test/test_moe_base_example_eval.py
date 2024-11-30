from typing import List

import pytest
import math
import numpy as np

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.multi_objective_evolution.nsga2_fitness import NSGA2Fitness
from examples.multi_objective.moe_base_example.nsga2_basic_example import (
    NSGA2BasicExampleEvaluator,
)


class TestNSGA2BasicExampleEvaluator:

    @classmethod
    def setup_class(self):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.evaluator = NSGA2BasicExampleEvaluator()

    @classmethod
    def teardown_class(self):
        """teardown any state that was previously setup with a call to
        setup_class.
        """
        pass

    def assert_vector_almost_equal(self, vec1, vec2, tolerance=1e-15):
        for val1, val2 in zip(vec1, vec2):
            assert val1 == pytest.approx(val2, tolerance)

    def evaluation_valid(self, my_list: List[float]) -> List[float]:
        n = 3  # use the instance with n = 3
        chromosome = np.array(my_list)
        ans = []
        ans.append(1 - math.exp(-sum((chromosome - 1 / math.sqrt(n)) ** 2)))
        ans.append(1 - math.exp(-sum((chromosome + 1 / math.sqrt(n)) ** 2)))
        return ans

    def test_eval_fixed_vals(self):
        ind = GAVectorCreator(
            length=3, bounds=(-4, 4), fitness_type=NSGA2Fitness, vector_type=FloatVector
        )
        ind.vector = [-1.04, 2.844, -2.54]
        actual = self.evaluator.evaluate_individual(ind)
        expected = [0.999, 0.999]
        self.assert_vector_almost_equal(actual, expected, 0.01)

    def test_eval_fixed_vals_2_first_half(self):
        ind = GAVectorCreator(
            length=3, bounds=(-4, 4), fitness_type=NSGA2Fitness, vector_type=FloatVector
        )
        ind.vector = [-3.5158362080315237, 0.06631935351665152, -2.921231400536878]
        actual = self.evaluator.evaluate_individual(ind)
        expected = [0.9999999999998029, 0.999999516777658]
        self.assert_vector_almost_equal(actual, expected, tolerance=0.000000000000001)

    def test_eval_fixed_5_vals_grate_values(self):
        ind = GAVectorCreator(
            length=3, bounds=(-4, 4), fitness_type=NSGA2Fitness, vector_type=FloatVector
        )
        ind.vector = [-3.4817449392563073, -2.728885522520228, 3.7399170615305213]
        actual = self.evaluator.evaluate_individual(ind)
        expected = [1.0, 0.9999999999999993]
        self.assert_vector_almost_equal(actual, expected, tolerance=0.000000000000001)

    def test_eval_fixed_vals_6_first_half(self):
        ind = GAVectorCreator(
            length=3, bounds=(-4, 4), fitness_type=NSGA2Fitness, vector_type=FloatVector
        )
        ind.vector = [-2.841398594290565, -1.4434725567566344, 3.6928271437413196]
        actual = self.evaluator.evaluate_individual(ind)
        expected = [0.9999999999913871, 0.9999999999661979]
        self.assert_vector_almost_equal(actual, expected, tolerance=0.000000000000001)

    def test_eval_fixed_vals_7_second_half(self):
        ind = GAVectorCreator(
            length=3, bounds=(-4, 4), fitness_type=NSGA2Fitness, vector_type=FloatVector
        )
        ind.vector = [3.768444634957887, -1.2647927144943072, 3.5023114497554335]
        actual = self.evaluator.evaluate_individual(ind)
        expected = [0.9999999997555407, 0.9999999999999998]
        self.assert_vector_almost_equal(actual, expected, tolerance=0.000000000000001)

    def test_eval_random_vals(self):
        random_tests_count = 10
        for i in range(random_tests_count):
            vector = list(np.random.uniform(-4, 4, 3))
            self.check_sigel_vector(vector)

    def check_sigel_vector(self, initial_vector):
        ind = GAVectorCreator(
            length=3, bounds=(-4, 4), fitness_type=NSGA2Fitness, vector_type=FloatVector
        )
        ind.vector = initial_vector
        actual = self.evaluator.evaluate_individual(ind)
        expected = self.evaluation_valid(initial_vector)
        self.assert_vector_almost_equal(actual, expected)

import random
import numpy as np
import itertools

from eckity.algorithms import SimpleEvolution
from eckity.evaluators import SimpleIndividualEvaluator
from eckity import Subpopulation


class RandomIndividualEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, ind):
        return random.random() + np.random.random()


class TestReproducibility:
    def test_reproducibility(self):
        # creates a Tree-GP experiment by default
        algo1 = SimpleEvolution(
            Subpopulation(RandomIndividualEvaluator(), population_size=10),
            random_seed=42,
            max_generation=10,
        )
        algo1.evolve()
        bor1 = algo1.best_of_run_

        # creates a Tree-GP experiment by default
        algo2 = SimpleEvolution(
            Subpopulation(RandomIndividualEvaluator(), population_size=10),
            random_seed=42,
            max_generation=10,
        )
        algo2.evolve()
        bor2 = algo2.best_of_run_

        xs = list(range(10))
        ys = list(range(10))
        zs = list(range(10))

        for x, y, z in itertools.product(xs, ys, zs):
            assert bor1.execute(x=x, y=y, z=z) == bor2.execute(x=x, y=y, z=z)

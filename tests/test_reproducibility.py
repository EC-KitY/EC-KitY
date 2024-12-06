import itertools
import random
from time import sleep

import numpy as np

from eckity.subpopulation import Subpopulation
from eckity.algorithms import SimpleEvolution
from eckity.base.untyped_functions import f_add, f_div, f_mul, f_sub
from eckity.creators import FullCreator
from eckity.evaluators import SimpleIndividualEvaluator
from eckity.genetic_operators import SubtreeCrossover, SubtreeMutation


class RandomIndividualEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, ind):
        return random.random() + np.random.random()


class TestReproducibility:
    def test_reproducibility(self):
        # creates a Tree-GP algorithm
        algo1 = SimpleEvolution(
            Subpopulation(
                RandomIndividualEvaluator(),
                population_size=10,
                creators=FullCreator(
                    function_set=[f_add, f_sub, f_mul, f_div],
                    terminal_set=["x", "y", "z"],
                ),
                operators_sequence=[SubtreeCrossover(), SubtreeMutation()],
            ),
            random_seed=42,
            max_generation=10,
        )
        algo1.evolve()
        bor1 = algo1.best_of_run_

        # creates a Tree-GP experiment by default
        algo2 = SimpleEvolution(
            Subpopulation(
                RandomIndividualEvaluator(),
                population_size=10,
                creators=FullCreator(
                    function_set=[f_add, f_sub, f_mul, f_div],
                    terminal_set=["x", "y", "z"],
                ),
                operators_sequence=[SubtreeCrossover(), SubtreeMutation()],
            ),
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

    def test_random_seeds(self):
        n_reps = 100
        seeds = []
        for i in range(n_reps):
            seed = SimpleEvolution(
                Subpopulation(
                    RandomIndividualEvaluator(),
                    creators=FullCreator(
                        function_set=[f_add, f_sub, f_mul, f_div],
                        terminal_set=["x", "y", "z"],
                    ),
                    operators_sequence=[SubtreeCrossover()],
                ),
            ).random_seed
            seeds.append(seed)

            # test fails when sleeping for 1e-324 seconds
            sleep(1e-323)

        assert len(set(seeds)) == n_reps

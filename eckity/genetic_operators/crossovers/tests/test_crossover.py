import random

import numpy as np

from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.ga.int_vector import IntVector
from eckity.genetic_encodings.gp.tree.tree_individual import Tree
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.vector_k_point_crossover \
    import VectorKPointsCrossover
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.fitness.simple_fitness import SimpleFitness


class TestCrossover:
    def test_vector_two_point_crossover(self):
        random.seed(0)
        length = 4
        v1 = IntVector(SimpleFitness(), length, bounds=(1, 10))
        v1.set_vector(list(range(1, 5)))
        v2 = IntVector(SimpleFitness(), length, bounds=(1, 10))
        v2.set_vector(list(range(5, 9)))

        # random sample will return [2, 3]
        expected_v1 = [5, 2, 7, 8]
        expected_v2 = [1, 6, 3, 4]

        crossover = VectorKPointsCrossover(k=2)
        crossover.apply_operator([v1, v2])
        assert v1.vector == expected_v1
        assert v2.vector == expected_v2
        assert v1.applied_operators == ['VectorKPointsCrossover']
        assert v2.applied_operators == ['VectorKPointsCrossover']

    def test_subtree_crossover(self):
        tree1 = Tree(fitness=GPFitness(fitness=1.0))
        tree2 = Tree(fitness=GPFitness(fitness=0.5))

        # Set the trees with known structures for easier verification
        tree1.tree = ['f_add', 'x0', 'x1']
        tree2.tree = ['f_sub', 'x0', 'x1']

        # Save the initial trees for comparison
        initial_tree = tree1.clone()

        # Set up SubtreeCrossover operator
        crossover = SubtreeCrossover()

        # Perform crossover
        crossover.apply_operator([tree1, tree2])

        # Assert that the crossover has been applied
        assert tree1.tree != initial_tree
        assert tree1.applied_operators == ['SubtreeCrossover']
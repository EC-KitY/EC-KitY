import pytest

from eckity.creators.gp_creators.full import FullCreator
from eckity.creators.gp_creators.grow import GrowCreator
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.functions import f_add, f_sub, f_mul, f_div
from eckity.genetic_encodings.gp.tree.tree_individual import Tree
from eckity.fitness.gp_fitness import GPFitness


@pytest.fixture
def ramped_hh_creator():
    # Initialize RampedHalfAndHalfCreator with appropriate parameters
    return RampedHalfAndHalfCreator(grow_creator=None,
                                    full_creator=None,
                                    init_depth=(2, 4),
                                    function_set=[f_add, f_sub, f_mul, f_div],
                                    terminal_set=['x', 'y', 'z', 0, 1, -1],
                                    erc_range=(-1.0, 1.0),
                                    bloat_weight=0.1)


class TestRampedHalfAndHalf:
    def test_create_individuals_depth_distribution(self, ramped_hh_creator):
        # Test the distribution of individuals across different depths
        n_individuals = 10  # Specify the number of individuals to create
        higher_is_better = False  # Specify the fitness direction
        created_individuals = ramped_hh_creator.create_individuals(n_individuals, higher_is_better)

        # Count the number of individuals for each depth
        depth_count = {}
        for ind in created_individuals:
            depth = ind.depth()
            if depth in depth_count:
                depth_count[depth] += 1
            else:
                depth_count[depth] = 1

        # Assertions
        min_depth, max_depth = ramped_hh_creator.init_depth
        for depth in range(min_depth, max_depth + 1):
            assert depth in depth_count  # Check if individuals are created for each depth
            assert depth_count[depth] > 0  # Check if there are non-zero individuals for each depth

    def test_create_individuals_sunny_day(self, ramped_hh_creator):
        # Test the create_individuals method with valid parameters (sunny day scenario)
        n_individuals = 10  # Specify the number of individuals to create
        higher_is_better = False  # Specify the fitness direction
        created_individuals = ramped_hh_creator.create_individuals(n_individuals, higher_is_better)

        # Assertions
        assert len(created_individuals) == n_individuals  # Check if correct number of individuals is created
        for ind in created_individuals:
            assert isinstance(ind, Tree)  # Check if each created individual is an instance of Tree class

    def test_create_individuals_rainy_day_invalid_params(self, ramped_hh_creator):
        # Test the create_individuals method with invalid parameters (rainy day scenario)
        n_individuals = -1  # Specify an invalid number of individuals (negative value)
        higher_is_better = False  # Specify the fitness direction

        # Call the create_individuals method with invalid parameters
        with pytest.raises(ValueError):
            ramped_hh_creator.create_individuals(n_individuals, higher_is_better)

    def test_create_individuals_edge_case(self, ramped_hh_creator):
        # Test the create_individuals method with edge case (empty population)
        n_individuals = 0  # Specify an empty population
        higher_is_better = False  # Specify the fitness direction

        # Call the create_individuals method with edge case
        created_individuals = ramped_hh_creator.create_individuals(n_individuals, higher_is_better)

        # Assertions
        assert len(created_individuals) == 0  # Check if no individuals are created for empty population

    def test_create_tree_full(self, ramped_hh_creator):
        # Test the create_tree method with valid parameters (sunny day scenario)
        tree_ind = Tree(fitness=GPFitness(fitness=0.5))  # Create a Tree individual with valid fitness object
        max_depth = 3  # Specify a valid maximum depth

        # Call the create_tree method
        ramped_hh_creator.init_method = ramped_hh_creator.full_creator
        ramped_hh_creator.create_tree(tree_ind, max_depth)

        # Assertions
        assert isinstance(tree_ind.tree, list)  # Check if tree is a list
        assert tree_ind.depth() <= max_depth  # Check if tree depth is within the specified maximum depth

    def test_create_tree_grow(self, ramped_hh_creator):
        # Test the create_tree method with valid parameters (sunny day scenario)
        tree_ind = Tree(fitness=GPFitness(fitness=0.5))  # Create a Tree individual with valid fitness object
        max_depth = 3  # Specify a valid maximum depth

        # Call the create_tree method
        ramped_hh_creator.init_method = ramped_hh_creator.grow_creator
        ramped_hh_creator.create_tree(tree_ind, max_depth)

        # Assertions
        assert isinstance(tree_ind.tree, list)  # Check if tree is a list
        assert tree_ind.depth() <= max_depth  # Check if tree depth is within the specified maximum depth


    def test_create_tree_rainy_day_invalid_max_depth(self, ramped_hh_creator):
        # Test the create_tree method with invalid maximum depth
        tree_ind = Tree(fitness=GPFitness(fitness=0.5))  # Create a Tree individual with valid fitness object
        invalid_max_depth = -1  # Specify an invalid maximum depth (negative value)

        ramped_hh_creator.init_method = ramped_hh_creator.grow_creator
        # Call the create_tree method
        with pytest.raises(ValueError):
            ramped_hh_creator.create_tree(tree_ind, invalid_max_depth)


if __name__ == "__main__":
    pytest.main()

import pytest

from eckity.creators.gp_creators.full import FullCreator
from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.gp.tree.functions import f_add, f_sub, f_mul, f_div
from eckity.genetic_encodings.gp.tree.tree_individual import Tree


@pytest.fixture
def full_creator():
    # Initialize FullCreator with appropriate parameters
    return FullCreator(init_depth=(2, 4), function_set=[f_add, f_sub, f_mul, f_div],
                       terminal_set=['x', 'y', 'z', 0, 1, -1], erc_range=(-1.0, 1.0), bloat_weight=0.1)

class TestFull:
    def test_create_individuals_sunny_day(self, full_creator):
        # Test the create_individuals method with valid parameters (sunny day scenario)
        n_individuals = 5  # Specify the number of individuals to create
        higher_is_better = False  # Specify the fitness direction
        created_individuals = full_creator.create_individuals(n_individuals, higher_is_better)

        # Assertions
        assert len(created_individuals) == n_individuals  # Check if correct number of individuals is created
        for ind in created_individuals:
            assert isinstance(ind, Tree)  # Check if each created individual is an instance of Tree class

    def test_invalid_parameters(self):
        # Test for invalid parameters provided to FullCreator
        with pytest.raises(Exception):
            # Providing invalid init_depth where min_depth > max_depth
            FullCreator(init_depth=(4, 2), function_set=[f_add, f_sub, f_mul, f_div],
                        terminal_set=['x', 'y', 'z', 0, 1, -1], erc_range=(-1.0, 1.0), bloat_weight=0.1)


    def test_create_individuals_rainy_day_invalid_params(self, full_creator):
        # Test the create_individuals method with invalid parameters (rainy day scenario)
        n_individuals = -1  # Specify an invalid number of individuals (negative value)
        higher_is_better = False  # Specify the fitness direction

        # Call the create_individuals method with invalid parameters
        with pytest.raises(ValueError):
            full_creator.create_individuals(n_individuals, higher_is_better)

    def test_create_tree_sunny_day(self, full_creator):
        # Test the create_tree method with valid parameters (sunny day scenario)
        tree_ind = Tree(fitness=GPFitness(fitness=0.5))  # Create a Tree individual with valid fitness object
        max_depth = 3  # Specify a valid maximum depth

        # Call the create_tree method
        full_creator.create_tree(tree_ind, max_depth)

        # Assertions
        assert isinstance(tree_ind.tree, list)  # Check if tree is a list
        assert tree_ind.depth() <= max_depth  # Check if tree depth is within the specified maximum depth

    def test_create_tree_rainy_day_invalid_max_depth(self, full_creator):
        # Test the create_tree method with invalid maximum depth
        tree_ind = Tree(fitness=GPFitness(fitness=0.5))  # Create a Tree individual with valid fitness object
        invalid_max_depth = -1  # Specify an invalid maximum depth (negative value)

        # Call the create_tree method
        with pytest.raises(ValueError):
            full_creator.create_tree(tree_ind, invalid_max_depth)

if __name__ == "__main__":
    pytest.main()

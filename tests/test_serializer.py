import unittest
import os

from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.full import FullCreator
from eckity.creators.gp_creators.grow import GrowCreator
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.tree_individual import Tree
from eckity.evaluators.stub_evaluator import StubEvaluator
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.population import Population
from eckity.serializer import Serializer, POPULATION_FILE_FORMAT
from eckity.subpopulation import Subpopulation


class TestSerializer(unittest.TestCase):
    def test_serialize_population(self):
        serializer = Serializer(population_file_name='test_serialize_population')
        fitness_func = StubEvaluator()
        sub_population = Subpopulation(fitness_func)
        population = Population([sub_population])
        population.create_population_individuals()

        # assert the population file doesn't exist in current directory before performing serialization
        self.assertFalse(os.path.isfile(
            f'{serializer.base_dir}/{serializer.population_file_name}.{POPULATION_FILE_FORMAT}'),
            msg=f'Population file should not exist before serialization in {self.test_serialize_population.__name__}'
        )

        serializer.serialize_population(population)

        # assert the population file doesn't exist in current directory after performing serialization
        self.assertTrue(os.path.isfile(
            f'{serializer.base_dir}/{serializer.population_file_name}.{POPULATION_FILE_FORMAT}'),
            msg=f'Population file should exist after serialization in {self.test_serialize_population.__name__}'
        )

        # remove the created file
        os.remove(f'{serializer.base_dir}/{serializer.population_file_name}.{POPULATION_FILE_FORMAT}')

    def test_serialize_deserialize_population(self):
        serializer = Serializer()
        fitness_func = StubEvaluator()
        sub_population = Subpopulation(fitness_func)
        population = Population(sub_population)
        population.create_population_individuals()

        serializer.serialize_population(population)
        deserialized_population = serializer.deserialize_population()

        os.remove(f'{serializer.base_dir}/{serializer.population_file_name}.{POPULATION_FILE_FORMAT}')
        # self.assertEqual(population, deserialized_population)
        self.assertEqual(population.__dict__, deserialized_population.__dict__)

    def test_get_obj_by_name_breeder(self):
        serializer = Serializer()
        events = ['test_event']

        static_breeder = SimpleBreeder(events=events)
        dynamic_breeder: SimpleBreeder = serializer.get_obj_by_name(
            SimpleBreeder.__module__,
            SimpleBreeder.__name__,
            events=events)

        # self.assertEqual(static_breeder, dynamic_breeder)
        self.assertEqual(static_breeder.__dict__, dynamic_breeder.__dict__)

    def test_get_obj_by_name_sub_population(self):
        serializer = Serializer()
        fitness_func = StubEvaluator()
        static_sub_population: Subpopulation = Subpopulation(fitness_func)

        dynamic_sub_population = serializer.get_obj_by_name(
            Subpopulation.__module__,
            Subpopulation.__name__,
            fitness_func
        )

        # TODO comparing dicts is problematic
        self.assertEqual(static_sub_population, dynamic_sub_population)
        # self.assertEqual(static_sub_population.__dict__, dynamic_sub_population.__dict__)

        # for k in static_sub_population.__dict__.keys():
        #     print('key = ', k)
        #     print('static value:', static_sub_population.__dict__[k])
        #     print('dynamic value:', dynamic_sub_population.__dict__[k])

    def test_get_obj_by_name_grow_gp_tree_creator(self):
        serializer = Serializer()

        init_depth = (3, 5)
        events = ['test_event']

        static_grow_gp_tree_creator = GrowCreator(init_depth, events=events)

        dynamic_grow_gp_tree_creator = serializer.get_obj_by_name(
            GrowCreator.__module__,
            GrowCreator.__name__,
            init_depth,
            events=events
        )

        # self.assertEqual(static_grow_gp_tree_creator, dynamic_grow_gp_tree_creator)
        self.assertEqual(static_grow_gp_tree_creator.__dict__, dynamic_grow_gp_tree_creator.__dict__)

    def test_get_obj_by_name_full_gp_tree_creator(self):
        serializer = Serializer()

        init_depth = (2, 4)
        events = ['test_event']

        static_full_gp_tree_creator = FullCreator(init_depth, events=events)

        dynamic_full_gp_tree_creator = serializer.get_obj_by_name(
            FullCreator.__module__,
            FullCreator.__name__,
            init_depth,
            events=events
        )

        self.assertEqual(static_full_gp_tree_creator, dynamic_full_gp_tree_creator)

    def test_get_obj_by_name_ramped_hh_gp_tree_creator(self):
        serializer = Serializer()

        init_depth = (4, 6)
        events = ['test_event']

        static_ramped_hh_gp_tree_creator = RampedHalfAndHalfCreator(init_depth, events=events)

        dynamic_ramped_hh_gp_tree_creator = serializer.get_obj_by_name(
            RampedHalfAndHalfCreator.__module__,
            RampedHalfAndHalfCreator.__name__,
            init_depth,
            events=events
        )

        self.assertEqual(static_ramped_hh_gp_tree_creator, dynamic_ramped_hh_gp_tree_creator)

    def test_get_obj_by_name_tree_individual(self):
        serializer = Serializer()

        fitness_func = StubEvaluator()
        init_depth = (3, 6)

        static_tree_individual = Tree(fitness_func, init_depth=init_depth)

        dynamic_tree_individual = serializer.get_obj_by_name(
            Tree.__module__,
            Tree.__name__,
            fitness_func,
            init_depth=init_depth
        )

        self.assertEqual(static_tree_individual, dynamic_tree_individual)

    def test_get_obj_by_name_tournament_selection(self):
        serializer = Serializer()

        tournament_size = 10
        hib = True
        events = ['test_event']

        static_tournament_selection = TournamentSelection(tournament_size, higher_is_better=hib, events=events)

        dynamic_tournament_selection = serializer.get_obj_by_name(
            TournamentSelection.__module__,
            TournamentSelection.__name__,
            tournament_size,
            higher_is_better=hib,
            events=events
        )

        self.assertEqual(static_tournament_selection, dynamic_tournament_selection)

    def test_get_obj_by_name_elitism_selection(self):
        serializer = Serializer()

        num_elites = 9
        hib = True
        events = ['test_event']

        static_elitism_selection = ElitismSelection(num_elites, higher_is_better=hib, events=events)

        dynamic_elitism_selection = serializer.get_obj_by_name(
            ElitismSelection.__module__,
            ElitismSelection.__name__,
            num_elites,
            higher_is_better=hib,
            events=events
        )

        self.assertEqual(static_elitism_selection, dynamic_elitism_selection)

    def test_get_obj_by_name_subtree_mutation(self):
        serializer = Serializer()

        num_mutated_individuals = 3
        events = ['test_event']

        static_subtree_mutation = SubtreeMutation(num_mutated_individuals, events=events)

        dynamic_subtree_mutation = serializer.get_obj_by_name(
            SubtreeMutation.__module__,
            SubtreeMutation.__name__,
            num_mutated_individuals,
            events=events
        )

        self.assertEqual(static_subtree_mutation, dynamic_subtree_mutation)


if __name__ == '__main__':
    unittest.main()

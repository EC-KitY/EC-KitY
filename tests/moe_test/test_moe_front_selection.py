import pytest

from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

from eckity.genetic_encodings.ga.float_vector import FloatVector

from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.multi_objective_evolution.nsga2_fitness import NSGA2Fitness
from eckity.multi_objective_evolution.nsga2_front_sorting import NSGA2FrontSorting

from eckity.population import Population
from eckity.subpopulation import Subpopulation


class FitnessIsVectorEval(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        """sets the fitness to be the value of the vector"""
        return individual.vector


class TestNSGA2FrontSelection:

    @classmethod
    def setup_class(self):
        """setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        self.selection = NSGA2FrontSorting()

    def _init_pop(self, values):
        self.sub_pop = Subpopulation(
            creators=GAVectorCreator(
                length=3,
                bounds=(-4, 4),
                fitness_type=NSGA2Fitness,
                vector_type=FloatVector,
            ),
            population_size=len(values),
            # user-defined fitness evaluation method
            evaluator=FitnessIsVectorEval(),
            # maximization problem (fitness is sum of values), so higher fitness is better
            higher_is_better=[True, True],
            elitism_rate=1 / 300,
            # genetic operators sequence to be applied in each generation
            operators_sequence=[],
            selection_methods=[
                # (selection method, selection probability) tuple
                (TournamentSelection(tournament_size=3, higher_is_better=True), 1)
            ],
        )

        self.pop = Population([self.sub_pop])
        individuals = []
        for vector in values:
            fitness = NSGA2Fitness(vector, higher_is_better=True)
            ind = FloatVector(
                fitness=fitness,
                bounds=(float("-inf"), float("inf")),
                length=len(vector),
            )
            ind.set_vector(vector)
            individuals.append(ind)
        self.sub_pop.individuals = individuals

    @classmethod
    def teardown_class(self):
        """teardown any state that was previously setup with a call to
        setup_class.
        """
        self.pop = None
        self.sub_pop = None

    def assert_vector_almost_equal(self, vec1, vec2, tolerance=0.01):
        for val1, val2 in zip(vec1, vec2):
            assert val1 == pytest.approx(val2, tolerance)

    def test_pareto_front_finding_pop_is_1(self):
        self._init_pop([[1, 1]])
        self.selection._init_domination_dict(self.sub_pop.individuals)
        pareto_front = self.selection._pareto_front_finding(self.sub_pop.individuals)
        assert pareto_front[0].fitness.fitness == [1, 1]

    def test_pareto_front_finding_pop_is_2(self):
        self._init_pop([[1, 1], [2, 2]])
        self.selection._init_domination_dict(self.sub_pop.individuals)
        pareto_front = self.selection._pareto_front_finding(self.sub_pop.individuals)
        assert len(pareto_front) == 1
        assert pareto_front[0].fitness.fitness == [2, 2]

    def test_pareto_front_finding_pop_is_151(self):
        self._init_pop([[k, k] for k in range(151)])
        self.selection._init_domination_dict(self.sub_pop.individuals)
        pareto_front = self.selection._pareto_front_finding(self.sub_pop.individuals)
        assert len(pareto_front) == 1
        assert pareto_front[0].fitness.fitness == [150, 150]

    def test_pareto_front_finding_front_size_3(self):
        pop = [[4, 1], [3, 2], [3, 3], [2, 3], [2, 4]]
        expected_front = [[4, 1], [3, 3], [2, 4]]
        self._init_pop(pop)
        self.selection._init_domination_dict(self.sub_pop.individuals)
        actual_pareto_front = self.selection._pareto_front_finding(
            self.sub_pop.individuals
        )
        self.check_same_front(actual_pareto_front, expected_front)

    def check_same_front(self, actual, expected):
        actual = [list(ind.fitness.fitness) for ind in actual]
        assert len(actual) == len(expected)
        actual = sorted(actual)
        expected = sorted(expected)
        expected = [list(x) for x in expected]
        assert all([a == b for a, b in zip(actual, expected)])

    def test_pareto_front_finding_front_size_3_float(self):
        pop = [[4.1, 1.1], [3.3, 2.5], [3.3, 3.3], [2, 3], [2, 4.5]]
        expected_front = [[4.1, 1.1], [3.3, 3.3], [2, 4.5]]
        self._init_pop(pop)
        self.selection._init_domination_dict(self.sub_pop.individuals)
        actual_pareto_front = self.selection._pareto_front_finding(
            self.sub_pop.individuals
        )
        self.check_same_front(actual_pareto_front, expected_front)

    def test_select_two_fronts(self):
        pop = [[4, 1], [3, 2], [3, 3], [2, 3], [2, 4], [2, 2], [1, 1]]
        expected_front = [[4, 1], [3, 3], [2, 4], [3, 2], [2, 3]]
        self._init_pop(pop)
        self.selection.select_for_population(self.pop, 5)
        self.check_same_front(self.sub_pop.individuals, expected_front)

    def test_select_for_population_simple_test(self):
        # Check that the function returns the expected pareto front for a simple case
        size = 4
        pop = [(1, 5), (2, 4), (3, 3), (4, 2)]
        self._init_pop(pop)
        self.selection.select_for_population(self.pop, size)
        self.check_same_front(
            self.sub_pop.individuals, [[1, 5], [2, 4], [3, 3], [4, 2]]
        )

    def test_select_for_population_single_indv(self):
        # Check that the function handles lists with a single point correctly
        size = 1
        pop = [(1, 5)]
        self._init_pop(pop)
        self.selection.select_for_population(self.pop, size)
        self.check_same_front(self.sub_pop.individuals, [[1, 5]])

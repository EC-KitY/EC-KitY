import pytest
from overrides import override

from eckity.subpopulation import Subpopulation
from eckity.algorithms import SimpleEvolution
from eckity.breeders import SimpleBreeder
from eckity.creators import FullCreator
from eckity.evaluators import SimpleIndividualEvaluator
from eckity.genetic_operators import GeneticOperator


class DummyIndividualEvaluator(SimpleIndividualEvaluator):
    @override
    def evaluate_individual(self, individual) -> float:
        return 0.0


class DummyCrossover(GeneticOperator):
    def __init__(self, probability=1.0, arity=2, events=None):
        self.individuals = None
        self.applied_individuals = None
        super().__init__(probability=probability, arity=arity, events=events)

    @override
    def apply(self, individuals):
        return individuals


def test_incompatible_operator_arities():
    breeder = SimpleBreeder()
    algo = SimpleEvolution(
        Subpopulation(
            DummyIndividualEvaluator(),
            creators=FullCreator(
                function_set=[lambda x: x],
                terminal_set=['x']
            ),
            population_size=10,
            operators_sequence=[DummyCrossover(arity=3)],
        ),
        breeder=breeder,
    )
    with pytest.raises(ValueError) as err_info:
        breeder.apply_breed(algo.population)
    assert "arity" in str(err_info)

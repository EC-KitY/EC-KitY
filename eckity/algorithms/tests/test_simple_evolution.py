from eckity import Subpopulation
from eckity.evaluators import SimpleIndividualEvaluator

from ..simple_evolution import SimpleEvolution


class DummyIndividualEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, ind):
        return 1


def test_initialize():
    # creates a Tree-GP experiment by default
    algo = SimpleEvolution(Subpopulation(DummyIndividualEvaluator()))
    algo.initialize()

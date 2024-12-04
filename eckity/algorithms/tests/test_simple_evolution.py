from eckity.subpopulation import Subpopulation
from eckity.evaluators import SimpleIndividualEvaluator
from eckity.creators import FullCreator
from eckity.genetic_operators.mutations.identity_transformation import (
    IdentityTransformation,
)

from ..simple_evolution import SimpleEvolution


class DummyIndividualEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, ind):
        return 1


def test_initialize():
    # creates a Tree-GP experiment by default
    algo = SimpleEvolution(
        Subpopulation(
            DummyIndividualEvaluator(),
            creators=FullCreator(
                function_set=[lambda x: x], terminal_set=["x"]
            ),
            operators_sequence=[IdentityTransformation()],
        )
    )
    algo.initialize()

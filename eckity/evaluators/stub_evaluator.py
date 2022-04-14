import random

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class StubEvaluator(SimpleIndividualEvaluator):
    def __init__(self):
        super().__init__()
        random.seed(0)

    def _evaluate_individual(self, individual):
        return random.random()

    def __eq__(self, other):
        return super().__eq__(other) and isinstance(other, StubEvaluator)

from itertools import product

import pandas as pd

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class KnapsackEvaluator(SimpleIndividualEvaluator):
    """
    Evaluator class for the Multiplexer problem, responsible of defining a fitness evaluation method and evaluating it

    Attributes
    -------

    """

    def __init__(self):
        super().__init__()

    def _evaluate_individual(self, individual):
        """
        Compute the fitness value of a given individual.

        Parameters
        ----------
        individual: Vector
            The individual to compute the fitness value for.

        Returns
        -------
        float
            The evaluated fitness value of the given individual.
            The value ranges from 0 (worst case) to 1 (best case).
        """
        return 0.0

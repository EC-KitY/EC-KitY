from itertools import product

import pandas as pd
import random

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

NUM_ITEMS = 20
MAX_WEIGHT = 50


class KnapsackEvaluator(SimpleIndividualEvaluator):
    """
    Evaluator class for the Multiplexer problem, responsible of defining a fitness evaluation method and evaluating it.
    In this example, fitness is a weighted average of the item price and the negative value of the item weight

    Attributes
    -------
    items: dict(int, tuple(int, float))
        dictionary of (item id: (weights, prices)) of the items

    w_weight: float, default=0.5
        weight of the item weight when calculating the fitness function

    w_price: float, default=0.5
        weight of the item price when calculating the fitness function
    """

    def __init__(self, w_weight=0.5, w_price=0.5):
        super().__init__()

        # Generate ramdom items for the problem (keys=weights, values=prices)
        self.items = {i: (random.randint(1, 10), random.uniform(0, 100)) for i in range(NUM_ITEMS)}
        self.w_weight = w_weight
        self.w_price = w_price

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
        weight, price = 0.0, 0.0
        for i in range(individual.size()):
            if individual.cell_value(i):
                weight += self.items[i][0]
                price += self.items[i][1]
        if weight > MAX_WEIGHT:
            return -1e6

        # price should be maximized and weight should be minimized, so fitness should be maximized
        return self.w_price * price - self.w_weight * weight

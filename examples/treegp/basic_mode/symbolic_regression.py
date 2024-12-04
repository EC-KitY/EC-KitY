"""
A simple example optimizing a three-variable function.
This is a non-sklearn setting so we use `evolve` and `execute`.
"""

import numpy as np
import pandas as pd

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.base.untyped_functions import *
from eckity.creators.gp_creators.half import HalfCreator
from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)
from eckity.genetic_operators.crossovers.subtree_crossover import (
    SubtreeCrossover,
)
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.statistics.best_average_worst_statistics import (
    BestAverageWorstStatistics,
)
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import (
    ThresholdFromTargetTerminationChecker,
)


class SymbolicRegressionEvaluator(SimpleIndividualEvaluator):
    """
    This module implements the Evaluator class, which delivers the fitness function.
    You will need to implement such a class to work with your own problem and fitness function.
    """

    def __init__(self):
        super().__init__()

        data = np.random.uniform(-100, 100, size=(200, 3))
        self.df = pd.DataFrame(data, columns=["x", "y", "z"])
        self.df["target"] = self.target_func(
            self.df["x"], self.df["y"], self.df["z"]
        )

    @staticmethod
    def target_func(x, y, z):
        """
        True regression function, the individuals
        Parameters
        ----------
        x, y, z: float
            Values to the parameters of the function.

        Returns
        -------
        float
            The result of target function activation.
        """
        return x + 2 * y + 3 * z

    def evaluate_individual(self, individual):
        """
        Parameters
        ----------
        individual : Tree
            An individual program tree in the gp population, whose fitness needs to be computed.
            Makes use of GPTree.execute, which runs the program.
            Calling `gptree.execute` must use keyword arguments that match the terminal-set variables.
            For example, if the terminal set includes `x` and `y` then the call is `gptree.execute(x=..., y=...)`.

        Returns
        -------
        float
            fitness value
        """
        x, y, z = self.df["x"], self.df["y"], self.df["z"]
        return np.mean(
            np.abs(individual.execute(x=x, y=y, z=z) - self.df["target"])
        )


def main():
    """
    Solve the Symbolic Regression problem by approximating the regression target function.

    Expected runtime: less than a minute (on 2 cores, 2.5 GHz CPU)
    Example of an optimal evolved tree:
    f_add
       f_add
          f_add
             f_add
                y
                z
             x
          z
       f_add
          y
          z
    """

    # each node of the GP tree is either a terminal or a function
    # function nodes, each has two children (which are its operands)
    function_set = [
        f_add,
        f_mul,
        f_sub,
        f_div,
        f_sqrt,
        f_log,
        f_abs,
        f_max,
        f_min,
        f_inv,
        f_neg,
    ]

    # terminal set, consisted of variables
    terminal_set = ["x", "y", "z"]

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(
            creators=HalfCreator(
                init_depth=(2, 4),
                terminal_set=terminal_set,
                function_set=function_set,
                erc_range=(-1.0, 1.0),
                bloat_weight=0.0001,
            ),
            population_size=200,
            # user-defined fitness evaluation method
            evaluator=SymbolicRegressionEvaluator(),
            # minimization problem (fitness is MAE), so higher fitness is worse
            higher_is_better=False,
            elitism_rate=0.05,
            # genetic operators sequence to be applied in each generation
            operators_sequence=[
                SubtreeCrossover(probability=0.9),
                SubtreeMutation(probability=0.2),
                ERCMutation(probability=0.05),
            ],
            selection_methods=[
                # (selection method, selection probability) tuple
                (
                    TournamentSelection(
                        tournament_size=4
                    ),
                    1,
                )
            ],
        ),
        max_workers=4,
        max_generation=40,
        termination_checker=ThresholdFromTargetTerminationChecker(
            optimal=0, threshold=0.001
        ),
        statistics=BestAverageWorstStatistics(),
    )

    # evolve the generated initial population
    algo.evolve()

    # execute the best individual after the evolution process ends, by assigning numeric values to the variable
    # terminals in the tree
    print(f"algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}")


if __name__ == "__main__":
    main()

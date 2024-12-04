from time import time
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.base.typed_functions import and2bools, or2bools, not2bools, if_then_else3bools
from eckity.creators.gp_creators.full import FullCreator
from eckity.genetic_operators.crossovers.subtree_crossover import (
    SubtreeCrossover,
)
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

from itertools import product
from numbers import Number

import pandas as pd
import numpy as np

from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)

NUM_SELECT_ENTRIES = 3
NUM_INPUT_ENTRIES = 2 ** NUM_SELECT_ENTRIES
NUM_COLUMNS = NUM_SELECT_ENTRIES + NUM_INPUT_ENTRIES
NUM_ROWS = 2 ** NUM_COLUMNS


def _target_func(s0, s1, s2, d0, d1, d2, d3, d4, d5, d6, d7):
    """
    Truth table of a 8-3 mux gate

    Returns the value of d_i, where i is the decimal value of the binary
    number gained from joining s0, s1 and s2 digits (see examples below).

    Parameters
    ----------
    s0-s2: bool
        select values for the mux gate

    d0-d7: bool
        input values for the mux gate

    Returns
    -------
    bool
        False or True, depends on the values of the given parameters

    Examples
    -------
    _target_func(s0=False, s1=False, s2=False, d0=True, ...) = True (value of input d0)
    _target_func(s0=False, s1=False, s2=True, d0=True, d1=False, ...) = False (value of input d1)
    """
    return (
            ((not s0) and (not s1) and (not s2) and d0)
            or ((not s0) and (not s1) and s2 and d1)
            or ((not s0) and s1 and (not s2) and d2)
            or ((not s0) and s1 and s2 and d3)
            or (s0 and (not s1) and (not s2) and d4)
            or (s0 and (not s1) and s2 and d5)
            or (s0 and s1 and (not s2) and d6)
            or (s0 and s1 and s2 and d7)
    )


class MuxEvaluator(SimpleIndividualEvaluator):
    """
    Evaluator class for the Multiplexer problem, responsible of defining
    a fitness evaluation method and evaluating it.

    Attributes
    -------
    inputs: pd.DataFrame
        Input columns representing all possible combinations
        of all select and input values.

    output: pandas Series
        All possible output values. Values depend on the matching rows
        from the inputs data frame.
    """

    def __init__(self):
        super().__init__()

        # construct a truth table of all combinations of ones and zeros
        values = [
            list(map(bool, x)) + [_target_func(*map(bool, x))]
            for x in product([False, True], repeat=_target_func.__code__.co_argcount)
        ]
        truth_tbl = pd.DataFrame(
            values,
            columns=(list(_target_func.__code__.co_varnames) + ["output"]),
        )

        # split dataframe to input columns and an output column
        self.inputs = truth_tbl.iloc[:, :-1]
        self.output = truth_tbl["output"]

    def evaluate_individual(self, individual):
        """
        Compute the fitness value of a given individual.

        Fitness evaluation is done calculating the accuracy between the tree
        execution result and the optimal result (multiplexer truth table).

        Parameters
        ----------
        individual: Tree
            The individual to compute the fitness value for.

        Returns
        -------
        float
            The evaluated fitness value of the given individual.
            The value ranges from 0 (worst case) to 1 (best case).
        """

        # select entries columns
        select_entries = {
            f"s{i}": self.inputs[f"s{i}"] for i in range(NUM_SELECT_ENTRIES)
        }
        # input entries columns
        input_entries = {
            f"d{i}": self.inputs[f"d{i}"] for i in range(NUM_INPUT_ENTRIES)
        }

        exec_res = individual.execute(**select_entries, **input_entries)

        if isinstance(exec_res, Number) or exec_res.shape == ():
            exec_res = np.full((NUM_ROWS,), bool(exec_res))

        # Normalize number of correct results
        return np.mean(exec_res == self.output)


def main():
    """
    The goal in the Multiplexer (Mux) problem to create a GP tree that approximates a Multiplexer logical gate.

    Expected run time: less than a minute (on 2 cores, 2.5 GHz CPU)
    Example output is provided at the end of the file.

    References
    ----------
    DEAP Multiplexer Example: https://deap.readthedocs.io/en/master/examples/gp_multiplexer.html
    """

    start_time = time()

    # The terminal set of the tree will contain the mux inputs (d0-d7 in a 8-3 mux gate),
    # 3 select lines (s0-s2 in a 8-3 mux gate) and the constants 0 and 1
    # Terminal set for boolean inputs
    typed_select_terminals = {f"s{i}": bool for i in range(NUM_SELECT_ENTRIES)}
    typed_input_terminals = {f"d{i}": bool for i in range(NUM_INPUT_ENTRIES)}
    terminal_set = {**typed_select_terminals, **typed_input_terminals}

    # Logical functions: and, or, not and if-then-else
    function_set = [and2bools, or2bools, not2bools, if_then_else3bools]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(
            creators=FullCreator(
                init_depth=(2, 4),
                terminal_set=terminal_set,
                function_set=function_set,
                bloat_weight=0.00001,
                root_type=bool,
            ),
            population_size=40,
            # user-defined fitness evaluation method
            evaluator=MuxEvaluator(),
            # this is a maximization problem (fitness is accuracy), so higher fitness is better
            higher_is_better=True,
            # genetic operators sequence to be applied in each generation
            operators_sequence=[
                SubtreeCrossover(probability=0.8, arity=2),
                SubtreeMutation(probability=0.1, arity=1),
            ],
            selection_methods=[
                # (selection method, selection probability) tuple
                (
                    TournamentSelection(
                        tournament_size=7, higher_is_better=True
                    ),
                    1,
                )
            ],
        ),
        max_workers=1,
        max_generation=40,
        termination_checker=ThresholdFromTargetTerminationChecker(
            optimal=1, threshold=0.01
        ),
        statistics=BestAverageWorstStatistics(),
        random_seed=10,
    )

    # evolve the generated initial population
    algo.evolve()

    # execute the best individual after the evolution process ends
    exec1 = algo.execute(
        s0=False, s1=False, s2=True, d0=False, d1=False, d2=True, d3=True,
        d4=True, d5=False, d6=False, d7=True
    )
    exec3 = algo.execute(
        s0=False, s1=True, s2=True, d0=False, d1=False, d2=True, d3=True,
        d4=True, d5=False, d6=False, d7=True
    )
    exec7 = algo.execute(
        s0=True, s1=True, s2=True, d0=False, d1=False, d2=True, d3=True,
        d4=True, d5=False, d6=False, d7=True
    )
    print("execute(s0=False, s1=True, s2=True, d1=False): expected = False, actual =", exec1)
    print("execute(s0=True, s1=True, s2=True, d3=True): expected = True, actual =", exec3)
    print("execute(s0=True, s1=True, s2=True, d7=True): expected = True, actual =", exec7)

    print("best pure fitness:", algo.best_of_run_.get_pure_fitness())

    print("total time:", time() - start_time)


if __name__ == "__main__":
    main()

"""
Example output (s - select inputs, d - data inputs):
f_if_then_else
    f_and
        f_not
            f_if_then_else
            f_or
                f_and
                    f_not
                        s1
                    f_if_then_else
                        f_not
                        f_or
                            s0
                            f_or
                                f_and
                                    s0
                                    d3
                                f_and
                                    d0
                                    s1
                        f_and
                        d1
                        s2
                        f_and
                        f_or
                            d4
                            f_and
                                f_not
                                    s1
                                f_if_then_else
                                    f_not
                                    f_or
                                        s0
                                        d5
                                    f_and
                                    d5
                                    f_if_then_else
                                        f_not
                                            d2
                                        f_not
                                            d4
                                        f_if_then_else
                                            d5
                                            d7
                                            d6
                                    f_and
                                    d1
                                    f_if_then_else
                                        d1
                                        s2
                                        d0
                        f_and
                            d5
                            d5
                f_and
                    f_if_then_else
                        f_if_then_else
                        s2
                        s2
                        s2
                        d7
                        d2
                    s1
            d7
            f_if_then_else
                f_and
                    f_if_then_else
                        d2
                        f_if_then_else
                        d3
                        d7
                        d7
                        f_if_then_else
                        d0
                        s1
                        d7
                    s1
                s1
                d7
        d7
    s1
    f_or
        f_and
            f_not
            s1
            f_if_then_else
            f_not
                f_or
                    s0
                    f_or
                        f_and
                        d4
                        d0
                        f_and
                        f_if_then_else
                            d2
                            f_if_then_else
                                d3
                                f_and
                                    f_or
                                    f_and
                                        d2
                                        f_and
                                            1
                                            d7
                                    d1
                                    f_if_then_else
                                    d5
                                    f_or
                                        d0
                                        f_and
                                            d0
                                            d4
                                    f_if_then_else
                                        f_if_then_else
                                            d0
                                            d0
                                            d7
                                        f_or
                                            d2
                                            d7
                                        s0
                                d7
                            f_if_then_else
                                d0
                                s1
                                d7
                        s1
            f_and
                d1
                f_if_then_else
                    s2
                    s2
                    d0
            f_and
                f_or
                    d4
                    f_and
                        f_not
                        s1
                        f_if_then_else
                        s2
                        s2
                        s2
                f_and
                    d5
                    d5
        f_and
            f_if_then_else
            f_if_then_else
                f_and
                    d5
                    d5
                s2
                s2
            d7
            d2
            s1
"""

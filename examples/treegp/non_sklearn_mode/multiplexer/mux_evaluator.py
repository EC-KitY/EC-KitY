from itertools import product
from numbers import Number

import pandas as pd
import numpy as np

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

NUM_SELECT_ENTRIES = 3
NUM_INPUT_ENTRIES = 2**NUM_SELECT_ENTRIES
NUM_COLUMNS = NUM_SELECT_ENTRIES + NUM_INPUT_ENTRIES
NUM_ROWS = 2**NUM_COLUMNS


def _target_func(s0, s1, s2, d0, d1, d2, d3, d4, d5, d6, d7):
    """
    Truth table of a 8-3 mux gate

    Returns the value of d_i, where i is the decimal value of the binary
    number gained from joining s0, s1 and s2 digits (see examples below).

    Parameters
    ----------
    s0-s2: int
        select values for the mux gate

    d0-d7: int
        input values for the mux gate

    Returns
    -------
    int
        0 (False) or 1 (True), depends on the values of the given parameters

    Examples
    -------
    _target_func(s0=0, s1=0, s2=0, d0=1, ...) = 1 (value of input d0)
    _target_func(s0=0, s1=0, s2=1, d0=1, d1=0, ...) = 0 (value of input d1)
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
            list(x) + [_target_func(*x)]
            for x in product([0, 1], repeat=_target_func.__code__.co_argcount)
        ]
        truth_tbl = pd.DataFrame(
            values,
            columns=(list(_target_func.__code__.co_varnames) + ["output"])
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

        # sometimes execute will return a single scalar (in constant trees)
        if isinstance(exec_res, Number) or exec_res.shape == ():
            exec_res = np.full((NUM_ROWS,), exec_res)

        # Return normalized number of correct results
        return np.mean(exec_res == self.output)

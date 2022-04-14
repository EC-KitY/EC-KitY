from itertools import product
from numbers import Number

import pandas as pd
import numpy as np

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

NUM_SELECT_ENTRIES = 3
NUM_INPUT_ENTRIES = 2 ** NUM_SELECT_ENTRIES
NUM_COLUMNS = NUM_SELECT_ENTRIES + NUM_INPUT_ENTRIES
NUM_ROWS = 2 ** NUM_COLUMNS


def _target_func(s0, s1, s2, d0, d1, d2, d3, d4, d5, d6, d7):
    """
    Truth table of a 8-3 mux gate

    Returns the value of d_i, where i is the decimal value of the binary number gained from joining s0, s1 and s2 digits
    (see examples below)

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
    _target_func(s0=0, s1=0, s2=0, d0=1, ...) = 1 (the value of input entry d0)
    _target_func(s0=0, s1=0, s2=1, d0=1, d1=0, ...) = 0 (the value of input entry d1)
    """
    return ((not s0) and (not s1) and (not s2) and d0) \
           or ((not s0) and (not s1) and s2 and d1) \
           or ((not s0) and s1 and (not s2) and d2) \
           or ((not s0) and s1 and s2 and d3) \
           or (s0 and (not s1) and (not s2) and d4) \
           or (s0 and (not s1) and s2 and d5) \
           or (s0 and s1 and (not s2) and d6) \
           or (s0 and s1 and s2 and d7)


class MuxEvaluator(SimpleIndividualEvaluator):
    """
    Evaluator class for the Multiplexer problem, responsible of defining a fitness evaluation method and evaluating it

    Attributes
    -------
    inputs: pandas DataFrame
        Input columns representing all possible combinations of all select values and input values.

    output: pandas Series
        All possible output values. Values depend on the matching rows from the inputs DataFrame.
    """

    def __init__(self):
        super().__init__()

        # construct a truth table of all combinations of ones and zeros
        values = [list(x) + [_target_func(*x)] for x in product([0, 1], repeat=_target_func.__code__.co_argcount)]
        truth_tbl = pd.DataFrame(values, columns=(list(_target_func.__code__.co_varnames) + ['output']))

        # split dataframe to input columns and an output column
        self.inputs = truth_tbl.iloc[:, :NUM_COLUMNS]
        self.output = truth_tbl['output']

    def _evaluate_individual(self, individual):
        """
        Compute the fitness value of a given individual.

        Fitness evaluation is done calculating the accuracy between the tree execution result and the optimal result
        (multiplexer truth table).

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
        s0, s1, s2 = self.inputs['s0'], self.inputs['s1'], self.inputs['s2']
        # input entries columns
        d0, d1, d2, d3, d4, d5, d6, d7 = self.inputs['d0'], self.inputs['d1'], self.inputs['d2'], self.inputs['d0'], \
                                         self.inputs['d4'], self.inputs['d5'], self.inputs['d6'], self.inputs['d7']

        exec_res = individual.execute(s0=s0, s1=s1, s2=s2, d0=d0, d1=d1, d2=d2, d3=d3, d4=d4, d5=d5, d6=d6, d7=d7)

        # sometimes execute will return a single scalar (in cases of constant trees)
        if isinstance(exec_res, Number) or exec_res.shape == np.shape(0):
            exec_res = np.full((NUM_ROWS,), exec_res)

        # The more "matches" the individual's execute result has with expected output,
        # the better the individual's fitness is.
        # Worst Case: the individual returned only wrong (binary) results, and should have a fitness of 0.
        # The bitwise_xor operator will return a vector of ones, which sums to NUM_ROWS, resulting in a fitness of 0.
        # Best case: the individual returned only right (binary) results, and should have a fitness of 1.
        # The bitwise_xor operator will return a vector of zeros, which sums to 0, resulting in a fitness of 1.
        return (NUM_ROWS - np.sum(np.bitwise_xor(exec_res, self.output))) / NUM_ROWS

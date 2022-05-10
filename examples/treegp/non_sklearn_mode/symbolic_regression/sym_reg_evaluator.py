"""
This module implements the fitness class, which delivers the fitness function.
You will need to implement such a class to work with your own problem and fitness function.
"""

import numpy as np
import pandas as pd

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


def _target_func(x, y, z):
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


class SymbolicRegressionEvaluator(SimpleIndividualEvaluator):
    """
    Compute the fitness of an individual.
    """

    def __init__(self):
        super().__init__()

        # np.random.seed(0)

        data = np.random.uniform(-100, 100, size=(200, 3))
        self.df = pd.DataFrame(data, columns=['x', 'y', 'z'])
        self.df['target'] = _target_func(self.df['x'], self.df['y'], self.df['z'])

    def _evaluate_individual(self, individual):
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
        x, y, z = self.df['x'], self.df['y'], self.df['z']
        return np.mean(np.abs(individual.execute(x=x, y=y, z=z) - self.df['target']))

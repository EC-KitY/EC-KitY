"""
This module implements the fitness evaluation class, which delivers the fitness function.
You will need to implement such a class to work with your own problem and fitness function.
"""
import numpy as np
from sklearn.metrics import accuracy_score

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator

CLASSIFICATION_THRESHOLD = 0


class ClassificationEvaluator(SimpleIndividualEvaluator):
    """
    Class to compute the fitness of an individual in classification problems.
    """

    def __init__(self, X=None, y=None):
        super().__init__()
        self.X = X
        self.y = y

    def set_context(self, context):
        """
        Receive X and y values and assign them to X and y fields.

        Parameters
        ----------
        context: tuple. first element is a numpy array of size (n_samples, n_features),
                        and the second element is a numpy array of size (n_samples, 1) or (n_samples,)
            X matrix and y vector, either (X_train, y_train) or (X_test, y_test), depending on the evolution stage

        Returns
        -------
        None.
        """
        self.X = context[0]
        self.y = context[1]

    def _evaluate_individual(self, individual):
        """
        Compute the fitness value by comparing the program tree execution result to the result vector y

        Parameters
        ----------
        individual: Tree
            An individual program tree in the GP population, whose fitness needs to be computed.
            Makes use of GPTree.execute, which runs the program.
            Calling `GPTree.execute` must use keyword arguments that match the terminal-set variables.
            For example, if the terminal set includes `x` and `y` then the call is `GPTree.execute(x=..., y=...)`.

        Returns
        -------
        float:
            computed fitness value
        """
        y_pred = self.classify_individual(individual)
        return accuracy_score(y_true=self.y, y_pred=y_pred)

    def classify_individual(self, individual):
        return np.where(individual.execute(self.X) > CLASSIFICATION_THRESHOLD, 1, 0)

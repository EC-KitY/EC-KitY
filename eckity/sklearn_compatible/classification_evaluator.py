"""
This module implements the fitness evaluation class, which delivers the fitness function.
You will need to implement such a class to work with your own problem and fitness function.
"""

import numpy as np
from sklearn.metrics import accuracy_score

from eckity.evaluators.simple_individual_evaluator import (
    SimpleIndividualEvaluator,
)
from scipy.special import expit as sigmoid

CLF_METHODS = ("sigmoid", "argmax", "softmax")


class ClassificationEvaluator(SimpleIndividualEvaluator):
    """
    Class to compute the fitness of an individual in classification problems.
    All simple classes assume only one sub-population.
    """

    def __init__(
        self,
        X=None,
        y=None,
        metric=accuracy_score,
        n_classes=2,
        clf_method=CLF_METHODS[0],
    ):
        super().__init__()
        self.X = X
        self.y = y
        self.metric = metric
        self.n_classes = n_classes

        if clf_method not in CLF_METHODS:
            raise ValueError(
                "classification method must be one of", CLF_METHODS
            )
        self.clf_method = clf_method

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

    def evaluate_individual(self, individual):
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
        return self.metric(y_true=self.y, y_pred=y_pred)

    def classify_individual(self, individual):
        clf_method_to_function = {
            "sigmoid": self._clf_sigmoid,
            "argmax": self._clf_argmax,
            "softmax": self._clf_softmax,
        }
        selected_func = clf_method_to_function[self.clf_method]
        return selected_func(individual)

    def _clf_sigmoid(self, individual):
        # normalize execute results between 0 and 1
        probs = sigmoid(individual.execute(self.X))
        # Create thresholds: 1/N, 2/N, ..., (N-1)/N
        thresholds = np.linspace(0, 1, self.n_classes + 1)[1:-1]
        return np.digitize(probs, thresholds)

    def _clf_argmax(self, individual):
        return self._clf_root_func(individual, "argmax")

    def _clf_softmax(self, individual):
        return self._clf_root_func(individual, "softmax")

    def _clf_root_func(self, individual, method):
        # assumes individual is a GP tree with argmax function in depth 1
        if method not in individual.root.function.__name__:
            raise ValueError(
                f"Individual must have {method} function in depth 0 to classify."
            )
        return individual.execute(self.X)

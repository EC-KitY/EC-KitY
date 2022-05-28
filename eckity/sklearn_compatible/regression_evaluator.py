"""
This module implements the fitness evaluation class, which delivers the fitness function.
You will need to implement such a class to work with your own problem and fitness function.
"""

from sklearn.metrics import mean_absolute_error

from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class RegressionEvaluator(SimpleIndividualEvaluator):
    """
    Computes the fitness of an individual in regression problems.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features), default=None
    Training/Test data.

    y: array-like of shape (n_samples,) or (n_samples, 1), default=None
    Target vector. used during the training phase.
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
            X matrix and y vector, either (X_train, y_train) or (X_test, None), depending on the evolution stage

        Returns
        -------
        None.

        Examples
        -------
        reg_eval = RegressionEvaluator()
        X, y = make_regression()
        X_train, X_test, y_train, y_test = train_test_split()
        reg_eval.set_context(X_train, y_train)
        """
        self.X = context[0]
        self.y = context[1]

    def _evaluate_individual(self, individual):
        """
        compute fitness value by computing the MAE between program tree execution result and y result vector

        Parameters
        ----------
        individual : Tree
            An individual program tree in the GP population, whose fitness needs to be computed.
            Makes use of GPTree.execute, which runs the program.
            In Sklearn settings, calling `individual.execute` must use a numpy array.
            For example, if self.X is X_train/X_test, the call is `individual.execute(self.X)`.

        Returns
        ----------
        float
            Computed fitness value - evaluated using MAE between the execution result of X and the vector y.
        """
        return mean_absolute_error(self.y, individual.execute(self.X))

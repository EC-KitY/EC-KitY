from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from eckity.sklearn_compatible.sklearn_wrapper import SklearnWrapper


class SKClassifier(SklearnWrapper, ClassifierMixin):
    def predict(self, X):
        """
        Compute output using best evolved individual.
        Use `predict` in a sklearn setting.
        Input is a numpy array.

        Parameters
        ----------
        X : array-like or sparse matrix of (num samples, num feautres)

        Returns
        -------
        y : array, shape (num samples,)
            Returns predicted values after applying classification.
        """

        # Check is fit had been called
        check_is_fitted(self)

        best_of_run_evaluator_ = self.algorithm.best_of_run_evaluator

        # y doesn't matter since we only need execute result and evolution has already finished
        best_of_run_evaluator_.set_context((X, None))

        return best_of_run_evaluator_.classify_individual(self.algorithm.best_of_run_)

    def predict_proba(self, X):
        raise NotImplementedError('not implemented yet')

    def predict_log_proba(self, X):
        raise NotImplementedError('not implemented yet')

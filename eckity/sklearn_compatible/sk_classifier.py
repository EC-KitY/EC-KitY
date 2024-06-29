from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from eckity.sklearn_compatible.sklearn_wrapper import SklearnWrapper
from eckity.sklearn_compatible.classification_evaluator import (
    ClassificationEvaluator,
)
from eckity.genetic_encodings.gp import Tree


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

        clf_eval: ClassificationEvaluator = (
            self.algorithm.get_individual_evaluator()
        )

        # ignore y since we only need execute result and evolution is finished
        clf_eval.set_context((X, None))

        return clf_eval.classify_individual(self.algorithm.best_of_run_)

    def predict_proba(self, X):
        raise NotImplementedError("not implemented yet")

    def predict_log_proba(self, X):
        raise NotImplementedError("not implemented yet")

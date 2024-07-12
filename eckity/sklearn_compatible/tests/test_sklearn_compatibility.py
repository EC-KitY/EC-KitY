"""
Tests for EC-KitY are currently under active development.
For now, this class is a placeholder for future tests to be added.
"""

import pytest
from sklearn.exceptions import NotFittedError

from eckity.algorithms import SimpleEvolution
from eckity.sklearn_compatible import ClassificationEvaluator
from eckity.sklearn_compatible import SKClassifier
from eckity.subpopulation import Subpopulation


class TestSklearnCompatibility:
    algo = SimpleEvolution(Subpopulation(evaluator=ClassificationEvaluator()))
    clf = SKClassifier(algo)

    def test_predict_without_fit(self):
        X = [1, 2, 3]
        with pytest.raises(NotFittedError):
            self.clf.predict(X)

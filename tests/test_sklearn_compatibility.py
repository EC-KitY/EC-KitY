"""
Tests for EC-KitY are currently under active development.
For now, this class is a placeholder for future tests to be added.
"""
import pytest
from sklearn.exceptions import NotFittedError

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator
from eckity.sklearn_compatible.sk_classifier import SKClassifier
from eckity.subpopulation import Subpopulation


class TestSklearnCompatibility:
    def test_predict_without_fit(self):
        algo = SimpleEvolution(Subpopulation(evaluator=ClassificationEvaluator()))
        clf = SKClassifier(algo)
        X = [1, 2, 3]
        with pytest.raises(NotFittedError):
            clf.predict(X)

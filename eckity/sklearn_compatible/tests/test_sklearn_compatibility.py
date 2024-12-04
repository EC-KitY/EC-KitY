"""
Tests for EC-KitY are currently under active development.
For now, this class is a placeholder for future tests to be added.
"""

import pytest
from sklearn.exceptions import NotFittedError

from eckity.subpopulation import Subpopulation
from eckity.algorithms import SimpleEvolution
from eckity.base.typed_functions import sqrt_float
from eckity.creators import GrowCreator
from eckity.genetic_operators import SubtreeCrossover, SubtreeMutation
from eckity.sklearn_compatible import ClassificationEvaluator, SKClassifier


class TestSklearnCompatibility:
    algo = SimpleEvolution(
        Subpopulation(
            evaluator=ClassificationEvaluator(),
            creators=GrowCreator(
                init_depth=(1, 1),
                function_set=[sqrt_float],
                terminal_set={"x": float},
            ),
            operators_sequence=[
                SubtreeCrossover(),
                SubtreeMutation(),
            ],
        )
    )
    clf = SKClassifier(algo)

    def test_predict_without_fit(self):
        X = [1, 2, 3]
        with pytest.raises(NotFittedError):
            self.clf.predict(X)

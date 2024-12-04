"""
Solving a sklearn_mode problem created through scikit-learn's `load breast cancer`.
This is an sklearn setting so we use `fit` and `predict`.
"""

from time import time

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eckity.subpopulation import Subpopulation
from eckity.algorithms import SimpleEvolution
from eckity.base.untyped_functions import (
    f_add,
    f_div,
    f_mul,
    f_sub,
)
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators import (
    SubtreeCrossover,
    SubtreeMutation,
    TournamentSelection,
)
from eckity.sklearn_compatible import ClassificationEvaluator, SKClassifier
from eckity.statistics import BestAverageWorstSizeTreeStatistics
from eckity.termination_checkers import ThresholdFromTargetTerminationChecker
from eckity.creators import HalfCreator


def main():
    """
    Evolve a GP Tree that classifies breast cancer,
    using a dataset from sklearn
    """
    start_time = time()

    # load the brest cancer dataset from sklearn
    X, y = load_breast_cancer(return_X_y=True)

    # Automatically generate a terminal set.
    # Since there are 30 features, set terminal_set to: ['x0', 'x1', ..., 'x29']
    terminal_set = create_terminal_set(X, typed=False)

    # Define function set
    function_set = [
        f_add,
        f_sub,
        f_mul,
        f_div,
    ]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(
            creators=HalfCreator(
                init_depth=(2, 4),
                terminal_set=terminal_set,
                function_set=function_set,
                bloat_weight=0.0001,
            ),
            population_size=100,
            evaluator=ClassificationEvaluator(metric=accuracy_score),
            # maximization problem (fitness is accuracy), so higher fitness is better
            higher_is_better=True,
            elitism_rate=0.05,
            # genetic operators sequence to be applied in each generation
            operators_sequence=[
                SubtreeCrossover(probability=0.9, arity=2),
                SubtreeMutation(probability=0.2, arity=1),
            ],
            selection_methods=[
                # (selection method, selection probability) tuple
                (
                    TournamentSelection(
                        tournament_size=4, higher_is_better=True
                    ),
                    1,
                )
            ],
        ),
        max_workers=1,
        max_generation=100,
        # optimal fitness is 1, evolution ("training") process will be finished when best fitness <= threshold
        termination_checker=ThresholdFromTargetTerminationChecker(
            optimal=1, threshold=0.03
        ),
        statistics=BestAverageWorstSizeTreeStatistics(),
    )
    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    classifier = SKClassifier(algo)

    # split brest cancer dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit the model (perform evolution process)
    classifier.fit(X_train, y_train)

    print("best individual:\n", algo.best_of_run_.root)

    # check training set results
    print(
        f"\nbest pure fitness over training set: {algo.best_of_run_.get_pure_fitness()}"
    )

    # check test set results by computing the accuracy score between the prediction result and the test set result
    test_score = accuracy_score(y_test, classifier.predict(X_test))
    print(f"test score: {test_score}")

    print(f"Total runtime: {time() - start_time} seconds.")


if __name__ == "__main__":
    main()

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import GridSearchCV

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.base.typed_functions import (
    abs_float,
    add2floats,
    div2floats,
    inv_float,
    log_float,
    max2floats,
    min2floats,
    mul2floats,
    neg_float,
    sqrt_float,
    sub2floats,
)
from eckity.creators.gp_creators.half import HalfCreator
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import (
    SubtreeCrossover,
)
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.sklearn_compatible.classification_evaluator import (
    ClassificationEvaluator,
)
from eckity.sklearn_compatible.sk_classifier import SKClassifier
from eckity.statistics.best_average_worst_statistics import (
    BestAverageWorstStatistics,
)
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import (
    ThresholdFromTargetTerminationChecker,
)

# Adding your own types and functions

t_argmax = type("argmax", (int,), {})


def argmax3(x0: float, x1: float, x2: float) -> t_argmax:
    return np.argmax([x0, x1, x2])


def main():
    """
    Demonstrate sklearn compatibility through use of grid search, solving a classification problem.
    """

    # load the brest cancer dataset from sklearn
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=3, n_informative=5
    )

    # Automatically generate a terminal set.
    # Since there are 10 features, set terminal_set to: ['x0', 'x1', ..., 'x9']
    terminal_set = create_terminal_set(X, typed=True)

    # Define function set
    function_set = [
        abs_float,
        add2floats,
        div2floats,
        inv_float,
        log_float,
        max2floats,
        min2floats,
        mul2floats,
        neg_float,
        sqrt_float,
        sub2floats,
    ]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(
            creators=HalfCreator(
                init_depth=(2, 4),
                terminal_set=terminal_set,
                function_set=function_set,
                bloat_weight=0.0001,
                root_type=float,
            ),
            population_size=10,
            evaluator=ClassificationEvaluator(),
            higher_is_better=True,
            elitism_rate=0.05,
            operators_sequence=[
                SubtreeCrossover(probability=0.9, arity=2),
                SubtreeMutation(probability=0.2, arity=1),
            ],
            selection_methods=[
                (
                    TournamentSelection(
                        tournament_size=4, higher_is_better=True
                    ),
                    1,
                )
            ],
        ),
        max_workers=4,
        max_generation=10,
        termination_checker=ThresholdFromTargetTerminationChecker(
            optimal=1, threshold=0.03
        ),
        statistics=BestAverageWorstStatistics(),
    )
    classifier = SKClassifier(algo)

    print("Showcasing GridSearchCV...")

    # Grid search parameters.
    # The Grid Search model will fit the classifier with each parameter value
    parameters = {"max_generation": [10, 20, 30]}

    model = GridSearchCV(classifier, parameters)
    model.fit(X, y)

    print(f"best params: {model.best_params_}")
    print(f"best score:  {model.best_score_}")


if __name__ == "__main__":
    main()

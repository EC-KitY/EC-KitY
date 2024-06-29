"""
Solving a sklearn_mode problem created through scikit-learn's `make_regression`.
This is an sklearn setting, so we use `fit` and `predict`.
"""

import numpy as np
from time import time

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.half import HalfCreator
from eckity.base.untyped_functions import untyped_add, untyped_mul, untyped_sub
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import (
    SubtreeCrossover,
)
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.sklearn_compatible.sk_regressor import SKRegressor
from eckity.statistics.best_average_worst_statistics import (
    BestAverageWorstStatistics,
)
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import (
    ThresholdFromTargetTerminationChecker,
)
from eckity.sklearn_compatible.regression_evaluator import RegressionEvaluator


# Adding your own functions
def untyped_add3(x1, x2, x3):
    return np.add(np.add(x1, x2), x3)


def untyped_mul3(x1, x2, x3):
    return np.multiply(np.multiply(x1, x2), x3)


def main():
    """
    Solve a regression problem imported from sklearn `make_regression` function, using GP Trees.
    Expected run time: ~25 minutes (on 2 cores, 2.5 GHz CPU)
    Example output (with an error of 0.09 on test set):
    untyped_sub
       untyped_mul
          x3
          57.9788
       untyped_sub
          untyped_mul
             untyped_add
                x1
                x0
             -69.8759
          untyped_add
             untyped_sub
                untyped_add
                   untyped_add
                      untyped_add
                         untyped_add
                            untyped_add
                               untyped_add
                                  x1
                                  x2
                               x2
                            untyped_add
                               untyped_add
                                  x2
                                  x2
                               untyped_sub
                                  x1
                                  x0
                         x1
                      untyped_add
                         untyped_add
                            untyped_add
                               untyped_sub
                                  untyped_sub
                                     x1
                                     untyped_mul
                                        x1
                                        -1.0477
                                  x1
                               x2
                            x4
                         x1
                   x1
                untyped_mul
                   -33.4406
                   untyped_add
                      x4
                      x4
             untyped_sub
                untyped_mul
                   x1
                   untyped_sub
                      x2
                      x2
                untyped_mul
                   untyped_add
                      x2
                      x0
                   untyped_sub
                      -19.0099
                      -1.0477
    """
    start_time = time()

    # generate a random regression problem
    X, y = make_regression(n_samples=500, n_features=5)

    # Automatically generate a terminal set.
    # Since there are 5 features, set terminal_set to: ['x0', 'x1', 'x2', 'x3', 'x4']
    terminal_set = create_terminal_set(X)

    # Set function set to binary addition, binary multiplication and binary subtraction
    function_set = [untyped_add, untyped_mul, untyped_sub]

    # Initialize Simple Evolutionary Algorithm instance
    algo = SimpleEvolution(
        Subpopulation(
            creators=HalfCreator(
                init_depth=(2, 4),
                terminal_set=terminal_set,
                function_set=function_set,
                bloat_weight=0.0001,
            ),
            population_size=1000,
            # user-defined fitness evaluation method
            evaluator=RegressionEvaluator(),
            # minimization problem (fitness is MAE), so higher fitness is worse
            higher_is_better=False,
            elitism_rate=0.05,
            # genetic operators sequence to be applied in each generation
            operators_sequence=[
                SubtreeCrossover(probability=0.9, arity=2),
                SubtreeMutation(probability=0.2, arity=1),
                ERCMutation(probability=0.05, erc_range=(-100, 100), arity=1),
            ],
            selection_methods=[
                # (selection method, selection probability) tuple
                (
                    TournamentSelection(
                        tournament_size=4, higher_is_better=False
                    ),
                    1,
                )
            ],
        ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=1000,
        # optimal fitness is 0, evolution ("training") process will be finished when best fitness <= threshold
        termination_checker=ThresholdFromTargetTerminationChecker(
            optimal=0, threshold=0.01
        ),
        statistics=BestAverageWorstStatistics(),
    )
    # wrap the simple evolutionary algorithm with sklearn-compatible regressor
    regressor = SKRegressor(algo)

    # split regression dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit the model (perform evolution process)
    regressor.fit(X_train, y_train)

    # check training set results
    print(
        f"\nBest pure fitness over training set: {algo.best_ountyped_run_.get_pure_fitness()}"
    )

    # check test set results by computing the MAE between the prediction result and the test set result
    test_score = mean_absolute_error(y_test, regressor.predict(X_test))
    print(f"Test score: {test_score}")

    print(f"Total runtime: {time() - start_time} seconds.")


if __name__ == "__main__":
    main()

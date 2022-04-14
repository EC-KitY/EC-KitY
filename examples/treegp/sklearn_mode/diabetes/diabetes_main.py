"""
Solving a sklearn_mode problem created through scikit-learn's `load breast cancer`.
This is an sklearn setting so we use `fit` and `predict`.
"""

from time import time

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.sklearn_compatible.sk_regressor import SkRegressor
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.functions import f_add, f_mul, f_sub, f_div, f_sqrt, f_log, f_abs, f_max, f_min, f_inv, \
    f_neg
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.erc_mutation import ErcMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from eckity.sklearn_compatible.regression_evaluator import RegressionEvaluator


def main():
    """
    Solve the Diabetes regression problem (dataset taken from sklearn) using evolutionary run
    """
    start_time = time()

    # set dataset, function set and terminal set
    X, y = load_diabetes(return_X_y=True)  # load diabetes dataset from sklearn
    terminal_set = create_terminal_set(X)  # Sets terminal_set to: ['x0', 'x1', 'x2', ..., 'x9']
    function_set = [f_add, f_mul, f_sub, f_div, f_sqrt, f_log, f_abs, f_neg, f_inv, f_max, f_min]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        erc_range=(-3500, 3500),
                                                        bloat_weight=0.0000001),
                      population_size=1000,
                      evaluator=RegressionEvaluator(),
                      higher_is_better=False,
                      elitism_rate=0.05,
                      operators_sequence=[
                          SubtreeCrossover(probability=0.9, arity=2),
                          SubtreeMutation(probability=0.2, arity=1),
                          ErcMutation(probability=0.05, arity=1)
                      ],
                      selection_methods=[
                          (TournamentSelection(tournament_size=100, higher_is_better=False), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=1000,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=0, threshold=0.1),
        statistics=BestAverageWorstStatistics()
    )
    regressor = SkRegressor(algo)

    # split dataset to training and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # transform the dataset using a sklearn StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train, y_train)  # scaled data has mean 0 and variance 1 (only over training set)
    X_test = sc.transform(X_test)  # use same scaler as one fitted to training data

    # fit the model (using evolution)
    regressor.fit(X_train, y_train)
    print(f'\nbest pure fitness over training set: {algo.best_of_run_.get_pure_fitness()}')

    # check the fitted model performance on test set
    test_score = mean_absolute_error(regressor.predict(X_test), y_test)
    print(f'test score: {test_score}')

    print(f'Total runtime: {time() - start_time} seconds.')


if __name__ == "__main__":
    main()

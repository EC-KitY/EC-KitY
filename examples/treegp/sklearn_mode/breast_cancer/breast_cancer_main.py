"""
Solving a sklearn_mode problem created through scikit-learn's `load breast cancer`.
This is an sklearn setting so we use `fit` and `predict`.
"""

from time import time
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.sklearn_compatible.sk_classifier import SKClassifier
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.functions import f_add, f_mul, f_sub, f_div, f_neg, f_sqrt, f_log, f_abs, f_inv, f_max, \
    f_min
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_avg_worst_size_tree_statistics import BestAverageWorstSizeTreeStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker

# Adding your own functions
from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator


def main():
    """
    Evolve a GP Tree that classifies breast cancer, using a dataset from sklearn

    Expected running time: ~6 minutes (on 2 cores, 2.5 GHz CPU)
    Example output (with over 92% accuracy on test set):
    f_div
       f_log
          x10
       f_min
          f_div
             f_log
                f_sqrt
                   f_sqrt
                      x7
             f_max
                x19
                x23
          f_div
             f_log
                f_mul
                   f_add
                      f_abs
                         f_neg
                            f_sqrt
                               x19
                      f_sub
                         f_inv
                            x23
                         x26
                   f_log
                      x4
             f_max
                f_add
                   f_abs
                      f_max
                         f_sub
                            f_inv
                               x23
                            x26
                         x23
                   f_add
                      x12
                      f_sqrt
                         f_sqrt
                            f_log
                               f_sqrt
                                  f_sqrt
                                     x7
                f_inv
                   f_mul
                      x19
                      x25
    """
    start_time = time()

    # load the brest cancer dataset from sklearn
    X, y = load_breast_cancer(return_X_y=True)

    # Automatically generate a terminal set.
    # Since there are 5 features, set terminal_set to: ['x0', 'x1', 'x2', ..., 'x9']
    terminal_set = create_terminal_set(X)

    # Define function set
    function_set = [f_add, f_mul, f_sub, f_div, f_sqrt, f_log, f_abs, f_neg, f_inv, f_max, f_min]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        bloat_weight=0.0001),
                      population_size=1000,
                      evaluator=ClassificationEvaluator(),
                      # maximization problem (fitness is accuracy), so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=0.05,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          SubtreeCrossover(probability=0.9, arity=2),
                          SubtreeMutation(probability=0.2, arity=1)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=1000,
        # optimal fitness is 1, evolution ("training") process will be finished when best fitness <= threshold
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=1, threshold=0.03),
        statistics=BestAverageWorstSizeTreeStatistics()
    )
    # wrap the basic evolutionary algorithm with a sklearn-compatible classifier
    classifier = SKClassifier(algo)

    # split brest cancer dataset to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # fit the model (perform evolution process)
    classifier.fit(X_train, y_train)

    # check training set results
    print(f'\nbest pure fitness over training set: {algo.best_of_run_.get_pure_fitness()}')

    # check test set results by computing the accuracy score between the prediction result and the test set result
    test_score = accuracy_score(y_test, classifier.predict(X_test))
    print(f'test score: {test_score}')

    print(f'Total runtime: {time() - start_time} seconds.')


if __name__ == "__main__":
    main()

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV

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
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from eckity.sklearn_compatible.classification_evaluator import ClassificationEvaluator


def main():
    """
    Demonstrate sklearn compatibility through use of grid search, solving the breast cancer problem.
    """

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
                      higher_is_better=True,
                      elitism_rate=0.05,
                      operators_sequence=[
                          SubtreeCrossover(probability=0.9, arity=2),
                          SubtreeMutation(probability=0.2, arity=1)
                      ],
                      selection_methods=[
                          (TournamentSelection(tournament_size=4, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=4,
        max_generation=1000,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=1, threshold=0.03),
        statistics=BestAverageWorstStatistics()
    )
    classifier = SKClassifier(algo)

    print('Showcasing GridSearchCV...')

    # Grid search parameters.
    # The Grid Search model will fit the classifier with every pair combination of the parameter lists
    # For example: (max_workers=1, max_generation=50), (max_workers=1, max_generation=100) and so on.
    parameters = {'max_workers': [1, 2, 4], 'max_generation': [50, 100, 150]}

    # create the grid search model and fit it several times, each time with a different combination of the parameters
    model = GridSearchCV(classifier, parameters)
    model.fit(X, y)

    print(f'best params: {model.best_params_}')
    print(f'best score:  {model.best_score_}')


if __name__ == "__main__":
    main()

from time import time

from sklearn.decomposition import PCA
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.sklearn_compatible.sk_regressor import SKRegressor
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.functions import f_add, f_mul, f_sub
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from eckity.sklearn_compatible.regression_evaluator import RegressionEvaluator


def main():
    """
    Demnstrate sklearn compatibility through use of pipeline.

    Sequentially apply a Pipeline of transforms with a final estimator (a regression algorithm in our case).
    In this example, we will use sklearn GridSearchCV to solve Symbolic Regression GP problem.
    """
    start_time = time()

    # generate a random regression problem
    X, y = make_regression(n_samples=500, n_features=5)

    # Automatically generate a terminal set.
    # Since there are 5 features, set terminal_set to: ['x0', 'x1', 'x2', 'x3', 'x4']
    terminal_set = create_terminal_set(X)

    # Set function set to binary addition, binary multiplication and binary subtraction
    function_set = [f_add, f_mul, f_sub]

    # Initialize Simple Evolutionary Algorithm instance
    algo = SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        erc_range=(-100, 100),
                                                        bloat_weight=0.0001),
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
                          ERCMutation(probability=0.05, arity=1)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=4, higher_is_better=False), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=1000,
        # optimal fitness is 0, evolution ("training") process will be finished when best fitness <= threshold
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=0, threshold=0.01),
        statistics=BestAverageWorstStatistics()
    )
    # wrap the simple evolutionary algorithm with sklearn compatible regressor
    regressor = SKRegressor(algo)

    # split dataset to 80% training set and 20% test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create a pipeline: first scale the dataset, then run PCA and finally perform the evolution stage
    print('Showcasing pipeline...')
    pipeline = make_pipeline(StandardScaler(), PCA(n_components=len(terminal_set)), regressor)
    pipeline.fit(X_train, y_train)

    # compute test score on test set
    test_score = mean_absolute_error(y_test, pipeline.predict(X_test))
    print(f'pipeline test score: {test_score}')

    print(f'Total runtime: {time() - start_time} seconds.')


if __name__ == '__main__':
    main()

from time import time


from sklearn.datasets import make_regression
from sklearn.model_selection import GridSearchCV

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.base.typed_functions import add2floats, mul2floats, sub2floats
from eckity.sklearn_compatible.sk_regressor import SKRegressor
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.half import HalfCreator
from eckity.base.untyped_functions import f_add, f_mul, f_sub
from eckity.genetic_encodings.gp.tree.utils import create_terminal_set
from eckity.genetic_operators.crossovers.subtree_crossover import (
    SubtreeCrossover,
)
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.statistics.best_average_worst_statistics import (
    BestAverageWorstStatistics,
)
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import (
    ThresholdFromTargetTerminationChecker,
)

from eckity.sklearn_compatible.regression_evaluator import RegressionEvaluator

# False for non-typed mode, True for strongly-typed mode
TYPED = False


def main():
    """
    Demonstrate sklearn compatibility through use of grid search

    Perform an exhaustive search over a given set of parameters to find the best parameters.
    In this example, we will use sklearn GridSearchCV to solve Symbolic Regression GP problem.

    Expected output:    best params: {'max_generation': 30}
    Expected runtime: ~16 minutes (on 2 cores, 2.5 GHz CPU)
    """
    start_time = time()

    # generate a random regression problem.
    X, y = make_regression(n_samples=500, n_features=5)

    # Automatically generate a terminal set.
    # Since there are 5 features, set terminal_set to: ['x0', 'x1', 'x2', 'x3', 'x4']
    terminal_set = create_terminal_set(X, typed=TYPED)

    # Set function set to binary addition, binary multiplication and binary subtraction
    untyped_function_set = [f_add, f_mul, f_sub]
    typed_function_set = [add2floats, mul2floats, sub2floats]
    function_set = typed_function_set if TYPED else untyped_function_set

    root_type = float if TYPED else None

    # Initialize Simple Evolutionary Algorithm instance
    algo = SimpleEvolution(
        Subpopulation(
            creators=HalfCreator(
                init_depth=(2, 4),
                terminal_set=terminal_set,
                function_set=function_set,
                bloat_weight=0.0001,
                erc_range=(-100.0, 100.0),
                root_type=root_type,
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
                ERCMutation(probability=0.05, arity=1),
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
    # wrap the simple evolutionary algorithm with sklearn compatible regressor
    regressor = SKRegressor(algo)

    print("Showcasing GridSearchCV...")

    # Grid search parameters.
    # The Grid Search model will fit the classifier with each parameter value
    parameters = {"max_generation": [10, 20, 30]}

    # create the grid search model and fit it several times, each time with a different combination of the parameters
    model = GridSearchCV(regressor, parameters)
    model.fit(X, y)

    print(f"best params: {model.best_params_}")
    print(f"Total runtime: {time() - start_time} seconds.")


if __name__ == "__main__":
    main()
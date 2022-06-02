"""
A simple example optimizing a three-variable function.
This is a non-sklearn setting so we use `evolve` and `execute`.
"""
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.ramped_hh import RampedHalfAndHalfCreator
from eckity.genetic_encodings.gp.tree.functions import f_add, f_mul, f_sub, f_div, \
    f_sqrt, f_log, f_abs, f_max, f_min, f_inv, f_neg
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from examples.treegp.non_sklearn_mode.symbolic_regression.sym_reg_evaluator import SymbolicRegressionEvaluator


def main():
    """
    Evolutionary experiment to create a GP tree that solves a Symbolic Regression problem
    In this example every GP Tree is a mathematical function.
    The goal is to create a GP Tree that produces the closest function to the regression target function
    """

    # each node of the GP tree is either a terminal or a function
    # function nodes, each has two children (which are its operands)
    function_set = [f_add, f_mul, f_sub, f_div, f_sqrt, f_log, f_abs, f_max, f_min, f_inv, f_neg]

    # terminal set, consisted of variables and constants
    terminal_set = ['x', 'y', 'z', 0, 1, -1]

    # Initialize the evolutionary algorithm
    algo = SimpleEvolution(
        Subpopulation(creators=RampedHalfAndHalfCreator(init_depth=(2, 4),
                                                        terminal_set=terminal_set,
                                                        function_set=function_set,
                                                        bloat_weight=0.0001),
                      population_size=200,
                      # user-defined fitness evaluation method
                      evaluator=SymbolicRegressionEvaluator(),
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
        max_workers=4,
        max_generation=500,
        # random_seed=0,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=0, threshold=0.001),
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()

    # execute the best individual after the evolution process ends, by assigning numeric values to the variable
    # terminals in the tree
    print(f'algo.execute(x=2,y=3,z=4): {algo.execute(x=2, y=3, z=4)}')


if __name__ == '__main__':
    main()

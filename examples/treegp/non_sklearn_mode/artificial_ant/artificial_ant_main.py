import numpy

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.full import FullCreator
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.erc_mutation import ErcMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.probabilistic_condition_operator import ProbabilisticConditionOperator
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_avg_worst_size_tree_statistics import BestAverageWorstSizeTreeStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from examples.treegp.non_sklearn_mode.artificial_ant.ant_simulator import AntSimulator
from examples.treegp.non_sklearn_mode.artificial_ant.ant_utills import prog2, prog3, move_forward
from examples.treegp.non_sklearn_mode.artificial_ant.artificial_ant_evaluator import ArtificialAntEvaluator

"""toolbox.register("evaluate", evalArtificialAnt)


hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("std", numpy.std)"""


def main():
    simulator = AntSimulator(600)
    with open("./santafe_trail.txt") as trail_file:
        simulator.parse_matrix(trail_file)

    terminal_set = ["turn_right", "turn_left", "move_forward"]

    food_ahead = simulator.if_food_ahead_helper()
    function_set = [food_ahead, prog2, prog3]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(creators=FullCreator(init_depth=(1, 2), terminal_set=terminal_set,
                                           function_set=function_set, bloat_weight=0.0025),
                      population_size=300,
                      evaluator=ArtificialAntEvaluator(simulator),
                      higher_is_better=True,
                      elitism_rate=0.03,
                      operators_sequence=[
                          SubtreeCrossover(probability=0.8, arity=2),
                          SubtreeMutation(probability=0.2, arity=1)
                      ],
                      selection_methods=[
                          (TournamentSelection(tournament_size=7, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=1000,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=1, threshold=0.01),
        statistics=BestAverageWorstSizeTreeStatistics()
    )

    algo.evolve()


if __name__ == "__main__":
    main()

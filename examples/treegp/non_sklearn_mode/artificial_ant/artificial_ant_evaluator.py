from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from examples.treegp.non_sklearn_mode.artificial_ant.ant_simulator import AntSimulator
from examples.treegp.non_sklearn_mode.artificial_ant.ant_utills import turn_right, turn_left, move_forward

MAX_PIECES_OF_FOOD = 89


class ArtificialAntEvaluator(SimpleIndividualEvaluator):
    def __init__(self, main_simulator=AntSimulator(600)):
        super().__init__()
        self.simulator = main_simulator

    def _evaluate_individual(self, individual):
        self.simulator.reset()
        individual.execute(turn_right=turn_right(self.simulator), turn_left=turn_left(self.simulator),
                           move_forward=move_forward(self.simulator))()
        return self.simulator.eaten / MAX_PIECES_OF_FOOD

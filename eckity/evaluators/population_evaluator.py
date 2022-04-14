from concurrent.futures.thread import ThreadPoolExecutor

from eckity.event_based_operator import Operator


class PopulationEvaluator(Operator):
    def __init__(self):
        super().__init__()
        self.executor = None

    def _evaluate(self, population):
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        population:
            a population instance

        Returns
        -------
        individual
            the individual with the best fitness out of the given individuals
        """
        self.applied_individuals = population

    def apply_operator(self, payload):
        return self._evaluate(payload)

    def set_executor(self, executor: ThreadPoolExecutor):
        self.executor = executor

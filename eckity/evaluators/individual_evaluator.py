from overrides import overrides

from eckity.event_based_operator import Operator


class IndividualEvaluator(Operator):

    def evaluate(self, individuals):
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        individuals:
            a list of individuals

        Returns
        -------
        individual
            the individual with the best fitness out of the given individuals
        """
        self.applied_individuals = individuals

    @overrides
    def apply_operator(self, payload):
        return self.evaluate(payload)

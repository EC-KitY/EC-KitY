from abc import abstractmethod


class TerminationChecker:
    """
    Abstract TerminationChecker class.

    This class is responsible of checking if the evolutionary algorithm should perform early termination.
    This class can be expanded depending on the defined termination condition.
    For example - threshold from target fitness, small change in fitness over a number of generations etc.
    """
    @abstractmethod
    def should_terminate(self, population, best_individual, gen_number):
        """
        Determines if the algorithm should perform early termination.

        Parameters
        ----------
        population: Population
            The population of the experiment.

        best_individual: Individual
            The best individual in the current generation of the algorithm.

        gen_number: int
            Current generation number.

        Returns
        -------
        bool
            True if the algorithm should terminate early, False otherwise.
        """
        pass

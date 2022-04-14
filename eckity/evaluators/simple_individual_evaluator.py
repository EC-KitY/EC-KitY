from abc import abstractmethod

from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator


class SimpleIndividualEvaluator(IndividualEvaluator):
    """
    Computes fitness value for the given individuals.
    In simple case, evaluates each individual separately.
    You will need to extend this class with your user-defined fitness evaluation methods.
    """
    @overrides
    def evaluate(self, individuals):
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        individuals: list of individuals
            individuals to evaluate fitness of  - in Simple Evaluator version the list is of size 1

        Returns
        -------
        Individual
            the individual with the best fitness out of the given individuals
        """
        assert len(individuals) == 1, 'SimpleIndividualEvaluator.evaluate must receive an individuals list of size 1'
        super().evaluate(individuals)
        individual = individuals[0]
        fitness_score = self._evaluate_individual(individual)
        individual.fitness.set_fitness(fitness_score)
        return individuals[0]

    @abstractmethod
    def _evaluate_individual(self, individual):
        """

        Parameters
        ----------
        individual: Individual
            The individual to compute the fitness for

        Returns
        -------
        float
            The evaluated fitness value for the given individual
        """
        pass

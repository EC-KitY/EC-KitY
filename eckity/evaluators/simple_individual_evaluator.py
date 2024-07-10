from abc import abstractmethod

from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator


class SimpleIndividualEvaluator(IndividualEvaluator):
    """
    Computes fitness value for the given individuals.
    All simple classes assume only one sub-population.
    Evaluates each individual separately.
    You will need to extend this class with your fitness evaluation methods.
    """

    @overrides
    def evaluate(self, individual, environment_individuals):
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        individual: Individual
                the current individual to evaluate its fitness

        environment_individuals: list of Individuals
                the individuals in the current individual's environment
                those individuals will affect the current individual's fitness
                (not used in simple case)

        Returns
        -------
        Individual
                the individual with the best fitness of the given individuals
        """
        super().evaluate(individual, environment_individuals)
        fitness_score = self.evaluate_individual(individual)
        individual.fitness.set_fitness(fitness_score)
        return individual

    @abstractmethod
    def evaluate_individual(self, individual):
        """
        Evaluate the fitness score for the given individual.
        This function must be implemented by subclasses of this class

        Parameters
        ----------
        individual: Individual
                The individual to compute the fitness for

        Returns
        -------
        float
                The evaluated fitness value for the given individual
        """
        raise ValueError(
            "evaluate_individual is an abstract method in SimpleIndividualEvaluator"
        )

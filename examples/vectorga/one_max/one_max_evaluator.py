from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator


class OneMaxEvaluator(SimpleIndividualEvaluator):
    def evaluate_individual(self, individual):
        """
            Compute the fitness value of a given individual.

            Parameters
            ----------
            individual: Vector
                The individual to compute the fitness value for.

            Returns
            -------
            float
                The evaluated fitness value of the given individual.
        """
        return sum(individual.vector)

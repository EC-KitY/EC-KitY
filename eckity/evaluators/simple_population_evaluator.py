from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual


class SimplePopulationEvaluator(PopulationEvaluator):
    @overrides
    def _evaluate(self, population):
        """
        Updates the fitness score of the given individuals, then returns the best individual

        Parameters
        ----------
        population:
            the population of the evolutionary experiment

        Returns
        -------
        individual
            the individual with the best fitness out of the given individuals
        """
        super()._evaluate(population)
        for sub_population in population.sub_populations:
            sp_eval: IndividualEvaluator = sub_population.evaluator
            eval_futures = [self.executor.submit(sp_eval.evaluate, [ind]) for ind in sub_population.individuals]

            # wait for all fitness values to be evaluated before returning from this method
            for future in eval_futures:
                future.result()

        # only one subpopulation in simple case
        individuals = population.sub_populations[0].individuals

        best_ind: Individual = population.sub_populations[0].individuals[0]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness

        return best_ind

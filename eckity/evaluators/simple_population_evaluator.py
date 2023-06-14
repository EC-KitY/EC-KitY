from overrides import overrides

from eckity.evaluators.individual_evaluator import IndividualEvaluator
from eckity.evaluators.population_evaluator import PopulationEvaluator
from eckity.fitness.fitness import Fitness
from eckity.individual import Individual


class SimplePopulationEvaluator(PopulationEvaluator):
    def __init__(self,
                 best_on: int = 0):
        super().__init__()
        self.best_on = best_on

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
            eval_results = self.executor.map(sp_eval.evaluate_individual, sub_population.individuals)
            for ind, fitness_score in zip(sub_population.individuals, eval_results):
                ind.fitness.set_fitness(fitness_score)

        assert self.best_on <= len(
            population.sub_populations), "Can't pick best solution on non-existing sub-population."
        individuals = population.sub_populations[self.best_on].individuals

        best_ind: Individual = population.sub_populations[self.best_on].individuals[self.best_on]
        best_fitness: Fitness = best_ind.fitness

        for ind in individuals[1:]:
            if ind.fitness.better_than(ind, best_fitness, best_ind):
                best_ind = ind
                best_fitness = ind.fitness

        return best_ind

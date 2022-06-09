

class Population:
    """
    Population of individuals to be evolved in the evolutionary run.

    Parameters
    ----------
    sub_populations: list of Subpopulations
        sub-populations contained in the population.
        For more information, see eckity.subpopulation.
    """
    def __init__(self, sub_populations):
        self.sub_populations = sub_populations

    def create_population_individuals(self):
        for sub_pop in self.sub_populations:
            sub_pop.create_subpopulation_individuals()

    def find_individual_subpopulation(self, individual):
        for sub_pop in self.sub_populations:
            if sub_pop.contains_individual(individual):
                return sub_pop
        raise ValueError('The given individual was not found in any sub-population.'
                         'It probably belongs to a previous generation population.')

    def get_best_individuals(self):
        return [sub_pop.get_best_individual() for sub_pop in self.sub_populations]

    def get_worst_individuals(self):
        return [sub_pop.get_worst_individual() for sub_pop in self.sub_populations]

    def get_average_fitness(self):
        return [sub_pop.get_average_fitness() for sub_pop in self.sub_populations]

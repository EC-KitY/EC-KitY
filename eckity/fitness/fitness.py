from abc import abstractmethod


class Fitness:
    """
    context : list of Individuals
        individuals involved in calculating the fitness (co-evolution)

    trials : list of floats
        fitness results for previous trials done to calculate fitness (co-evolution)
    """
    def __init__(self,
                 context=None,
                 trials=None,
                 is_evaluated=False,
                 is_relative_fitness=False,
                 should_cache_between_gens=False,
                 higher_is_better=False):
        self.context = context
        self.trials = trials
        self._is_evaluated = is_evaluated
        self.is_relative_fitness = is_relative_fitness
        self.should_cache_between_gens = False if is_relative_fitness else should_cache_between_gens
        self.higher_is_better = higher_is_better
        self.optimal_fitness = 1 if higher_is_better else 0

    @abstractmethod
    def get_pure_fitness(self):
        pass

    def get_augmented_fitness(self, individual):
        return self.get_pure_fitness()

    @abstractmethod
    def better_than(self, ind, other_fitness, other_ind):
        pass

    @abstractmethod
    def equal_to(self, ind, other_fitness, other_ind):
        pass

    def set_not_evaluated(self):
        if not self.is_fitness_evaluated():
            raise ValueError('Fitness already not evaluated')
        self._is_evaluated = False

    def is_fitness_evaluated(self):
        if self.is_relative_fitness:
            return True
        return self._is_evaluated

from overrides import overrides

from eckity.fitness.fitness import Fitness


class SimpleFitness(Fitness):
    def __init__(self,
                 fitness: float = None,
                 higher_is_better=False):
        is_evaluated = fitness is not None
        super().__init__(higher_is_better=higher_is_better, is_evaluated=is_evaluated)
        self.fitness: float = fitness

    def set_fitness(self, fitness):
        if self._is_evaluated:
            raise AttributeError('fitness already evaluated and set to', self.fitness)
        self.fitness = fitness
        self._is_evaluated = True

    @overrides
    def get_pure_fitness(self):
        if not self._is_evaluated:
            raise ValueError('Fitness not evaluated yet')
        return self.fitness

    @overrides
    def set_not_evaluated(self):
        super().set_not_evaluated()
        self.fitness = None

    def check_comparable_fitnesses(self, other_fitness):
        if not isinstance(other_fitness, SimpleFitness):
            raise TypeError('Expected SimpleFitness object in better_than, got', type(other_fitness))
        if not self.is_fitness_evaluated() or not other_fitness.is_fitness_evaluated():
            raise ValueError('Fitnesses must be evaluated before comparison')

    def better_than(self, ind, other_fitness, other_ind):
        self.check_comparable_fitnesses(other_fitness)
        return self.get_augmented_fitness(ind) > other_fitness.get_augmented_fitness(other_ind) \
            if self.higher_is_better \
            else self.get_augmented_fitness(ind) < other_fitness.get_augmented_fitness(other_ind)

    def equal_to(self, ind, other_fitness, other_ind):
        self.check_comparable_fitnesses(other_fitness)
        return self.get_augmented_fitness(ind) == other_fitness.get_augmented_fitness(other_ind)

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.should_cache_between_gens:
            state['_is_evaluated'] = False
            state['fitness'] = None
        return state

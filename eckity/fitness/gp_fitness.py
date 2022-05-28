from overrides import overrides

from eckity.fitness.simple_fitness import SimpleFitness


class GPFitness(SimpleFitness):
    def __init__(self,
                 fitness: float = None,
                 higher_is_better=False,
                 bloat_weight=0.1):
        super().__init__(fitness=fitness, higher_is_better=higher_is_better)
        self.bloat_weight = bloat_weight

    @overrides
    def get_augmented_fitness(self, individual):
        if not self.is_fitness_evaluated():
            raise ValueError('Fitness not evaluated yet')
        return self.fitness - self.bloat_weight * individual.size() \
            if self.higher_is_better \
            else self.fitness + self.bloat_weight * individual.size()

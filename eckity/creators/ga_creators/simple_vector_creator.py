from eckity.creators.creator import Creator
from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class GAVectorCreator(Creator):
    def __init__(self,
                 length=1,
                 gene_creator=None,
                 bounds=(0.0, 1.0),
                 vector_type=BitStringVector,
                 events=None):
        if events is None:
            events = ["after_creation"]
        super().__init__(events)

        if gene_creator is None:
            gene_creator = self.default_gene_creator
        self.gene_creator = gene_creator

        self.type = vector_type
        self.length = length
        self.bounds = bounds

    def create_individuals(self, n_individuals, higher_is_better):
        individuals = [self.type(length=self.length,
                                 bounds=self.bounds,
                                 fitness=SimpleFitness(higher_is_better=higher_is_better))
                       for _ in range(n_individuals)]
        for ind in individuals:
            self.create_vector(ind)
        self.created_individuals = individuals

        return individuals

    def create_vector(self, individual):
        # vector = [self.gene_creator(individual.bounds[i % len(individual.bounds)]) for i in individual.size()]
        vector = [self.gene_creator(individual, i) for i in range(self.length)]
        individual.set_vector(vector)

    @staticmethod
    def default_gene_creator(individual, index):
        return individual.get_random_number_in_bounds(index)

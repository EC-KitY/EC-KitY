from eckity.creators.creator import Creator
from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga.vector_individual import Vector

VECTOR_TYPES = {
            "bit_string_vector": BitStringVector,
            "int_vector": IntVector,
            "float_vector": FloatVector
        }


class GAVectorCreator(Creator):
    def __init__(self,
                 length=1,
                 gene_creator=None,
                 vector_type="bit_string_vector",
                 events=None):
        if events is None:
            events = ["after_creation"]
        super().__init__(events)

        if gene_creator is None:
            gene_creator = lambda individual, index: individual.get_random_number_in_bounds(index)
        self.length = length
        self.gene_creator = gene_creator

        self.type = VECTOR_TYPES[vector_type]

    def create_individuals(self, n_individuals, higher_is_better):
        individuals = [self.type(length=self.length,
                              cell_range=self.cell_range,
                              fitness=SimpleFitness(higher_is_better=higher_is_better))
                       for _ in range(n_individuals)]
        for ind in individuals:
            self.create_vector(ind)
        self.created_individuals = individuals

        return individuals

    def create_vector(self, individual):
        # vector = [self.gene_creator(individual.bounds[i % len(individual.bounds)]) for i in individual.size()]
        vector = [self.gene_creator(individual, i) for i in individual.size()]
        individual.set_vector(vector)

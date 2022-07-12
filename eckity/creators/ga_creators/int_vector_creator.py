from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.int_vector import IntVector


class GAIntVectorCreator(GAVectorCreator):
    def __init__(self,
                 length=1,
                 gene_creator=None,
                 events=None):
        super().__init__(length, gene_creator, IntVector, events)

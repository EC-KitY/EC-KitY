from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.float_vector import FloatVector


class GAFloatVectorCreator(GAVectorCreator):
    def __init__(self,
                 length=1,
                 gene_creator=None,
                 events=None):
        super().__init__(length, gene_creator, FloatVector, events)

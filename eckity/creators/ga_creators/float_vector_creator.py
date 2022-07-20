from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.float_vector import FloatVector


class GAFloatVectorCreator(GAVectorCreator):
    def __init__(self,
                 length=1,
                 gene_creator=None,
                 bounds=(0.0, 1.0),
                 events=None):
        super().__init__(length=length,
                         gene_creator=gene_creator,
                         bounds=bounds,
                         vector_type=FloatVector,
                         events=events)

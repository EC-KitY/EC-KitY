from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class GABitStringVectorCreator(GAVectorCreator):
    def __init__(self,
                 length=1,
                 bounds=(0, 1),
                 gene_creator=None,
                 events=None):
        super().__init__(length=length, bounds=bounds, gene_creator=gene_creator, vector_type=BitStringVector,
                         events=events)

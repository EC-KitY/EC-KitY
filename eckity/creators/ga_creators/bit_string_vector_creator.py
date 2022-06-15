from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class GABitStringVectorCreator(GAVectorCreator):
    def __init__(self,
                 fitness,
                 length=1,
                 bounds=(0.0, 1.0),
                 gene_creator=None,
                 events=None):
        super().__init__(fitness,length=length,bounds=bounds,gene_creator=gene_creator,vector_type=BitStringVector,events=events)
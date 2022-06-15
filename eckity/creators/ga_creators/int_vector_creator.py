
class GAIntVectorCreator(GAVectorCreator):
    def __init__(self,
                 length=1,
                 gene_creator=None,
                 events=None):
        super().__init__(length,gene_creator,IntVector,events)
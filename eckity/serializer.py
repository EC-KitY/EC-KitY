import pickle as pkl

from eckity.population import Population

POPULATION_FILE_FORMAT = "pkl"


class Serializer:
    def __init__(self,
                 base_dir=".",
                 population_file_name="population"):
        self.base_dir = base_dir
        self.population_file_name = population_file_name

    def serialize_population(self, population: Population):
        with open(f'{self.base_dir}/{self.population_file_name}.{POPULATION_FILE_FORMAT}', 'wb') as population_file:
            pkl.dump(population, population_file)

    def deserialize_population(self):
        with open(f'{self.base_dir}/{self.population_file_name}.{POPULATION_FILE_FORMAT}', 'rb') as population_file:
            return pkl.load(population_file)

    def get_obj_by_name(self, class_path, class_name, *args, **kwargs):
        # dynamically import the class module
        m = __import__(class_path, globals(), locals(), [class_name])
        # getattr returns the class, then we invoke the init method of the class with args and kwargs
        return getattr(m, class_name)(*args, **kwargs)

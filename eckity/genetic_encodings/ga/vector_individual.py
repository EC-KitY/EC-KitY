"""
This module implements the vector class.
"""

from random import randint

from eckity.individual import Individual


class Vector(Individual):
    def __init__(self,
                 fitness,
                 bounds,
                 length=1):
        super().__init__(fitness)

        # TODO do we need this assertion? @tomer
        #assert (type(bounds) == tuple and len(bounds == 2)) or (type(bounds) == list and len(bounds) == length)

        self.bounds = bounds
        self.length = length
        self.vector = []

    def size(self):
        """
        Compute size of vector.

        Returns
        -------
        int
            vector size (= number of cells).
        """
        return len(self.vector)

    def get_bounds(self):
        return self.bounds

    def check_if_in_bounds(self):
        for i in range(self.size()):
            if len(self.bounds) == 2:
                if (self.vector[i] < self.bounds[0]) | (self.vector[i] > self.bounds[1]):
                    return False
            else:
                if (self.vector[i] < self.bounds[i][0]) | (self.vector[i] > self.bounds[i][1]):
                    return False
        return True

    def add_cell(self, cell):
        self.vector.append(cell)

    def empty_vector(self):
        self.vector = []

    def set_vector(self, vector):
        self.vector = vector

    def get_vector(self):
        return self.vector

    def random_vector_part(self):
        # todo add tests to make sure this logic works
        rnd_i = randint(0, self.size() - 1)
        end_i = randint(rnd_i, self.size() - 1)
        return self.vector[rnd_i:end_i + 1]

    def replace_vector_part_random(self, inserted_part):
        index = randint(0, self.size() - len(inserted_part))  # select a random index
        end_i = index + len(inserted_part)
        replaced_part = self.vector[index:end_i]
        self.vector = self.vector[:index] + inserted_part + self.vector[end_i:]
        return replaced_part

    def replace_vector_part(self, inserted_part, start_index):
        end_i = start_index + len(inserted_part)
        replaced_part = self.vector[start_index:end_i]
        self.vector = self.vector[:start_index] + inserted_part + self.vector[end_i:]
        return replaced_part

    def get_vector_part(self, index, end_i):
        return self.vector[index:end_i]

    def cell_value(self, index):
        return self.vector[index]

    def set_cell_value(self, index, value):
        self.vector[index] = value

    def get_random_number_in_bounds(self, index):
        raise ValueError("get_random_number is abs method in vector individual")

    def show(self):
        """
        Print out a simple textual representation of the vector.

        Returns
        -------
        None.
        """
        print(self.vector)

# end class Vector

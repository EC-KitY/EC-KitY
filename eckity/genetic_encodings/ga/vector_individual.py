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

        assert (type(bounds) == tuple and len(bounds == 2)) or (type(bounds) == list and len(bounds) == length)
        self.bounds = bounds  # todo remove cell_range
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

    def replace_vector_part_random(self, vector):
        """
        Replace the subtree starting at `index` with `vector`

        Parameters
        ----------
        subtree - new subtree to replace the some existing subtree in this individual's tree

        Returns
        -------
        None
        """
        index = randint(0, self.size() - len(vector))  # select a random node (index)
        end_i = index + len(vector)
        replaced_part = self.vector[index:end_i]
        # todo add a test to make sure this logic works
        self.vector = self.vector[index:].extend(vector).extend(self.vector[:end_i])
        return replaced_part

    def replace_vector_part(self, vector, start_index):
        """
        Replace the subtree starting at `index` with `vector`

        Parameters
        ----------
        subtree - new subtree to replace the some existing subtree in this individual's tree

        Returns
        -------
        None
        """
        end_i = start_index + len(vector)
        replaced_part = self.vector[start_index:end_i]
        # todo add a test to make sure this logic works
        self.vector = self.vector[start_index:].extend(vector).extend(self.vector[:end_i])  # todo check
        return replaced_part

    def get_vector_part(self, index, end_i):
        return self.vector[index:end_i]

    def _cell_value(self, index):
        return self.vector[index]

    def set_cell_value(self, index, value):
        self.vector[index] = value

    def get_random_number_in_bounds(self, index):
        raise Exception("get_random_number is abs method in vector individual")

    def show(self):
        """
        Print out a simple textual representation of the vector.

        Returns
        -------
        None.
        """
        print(self.vector)

# end class tree

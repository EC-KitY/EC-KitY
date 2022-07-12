"""
This module implements the vector class.
"""
from abc import abstractmethod
from random import randint

from eckity.individual import Individual


class Vector(Individual):
    """
    A Vector individual representation for Genetic Algorithms operations.
    It is represented by a list of values (integers, floats, etc.)

    Parameters
    ----------
    fitness : Fitness
        Fitness handler class, responsible of keeping the fitness value of the individual.

    length : int
        Vector length - the number of cells in the vector.

    bounds : list of tuples
        Min/Max values for each vector cell (if of length n), or the minimum and maximum (if of length 1).
    """
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
        """
        Get vector bounds

        Returns
        -------
        tuple of (Number, Number)
            vector bounds.
        """
        return self.bounds

    def check_if_in_bounds(self):
        """
        Check if all vector cells are in bounds

        Returns
        -------
        bool
            True if all vector cells are in bounds, False otherwise
        """
        for i in range(self.size()):
            if len(self.bounds) == 2:
                if (self.vector[i] < self.bounds[0]) or (self.vector[i] > self.bounds[1]):
                    return False
            else:
                if (self.vector[i] < self.bounds[i][0]) or (self.vector[i] > self.bounds[i][1]):
                    return False
        return True

    def add_cell(self, cell):
        """
        Add a new cell to the vector (and increase its size by 1)

        Returns
        -------
        None
        """
        self.vector.append(cell)
        self.length += 1

    def empty_vector(self):
        """
        Convert the vector to an empty vector

        Returns
        -------
        None
        """
        self.vector = []
        self.length = 0

    def set_vector(self, vector):
        """
        Set genome to the given vector genome

        Parameters
        -------
        vector: list
            `other` vector genome

        Returns
        -------
        None
        """
        self.vector = vector
        self.length = len(vector)

    def get_vector(self):
        """
        Return self vector genome

        Returns
        -------
        list
            vector genome
        """
        return self.vector

    def random_vector_part(self):
        """
        Get a random part of the vector

        Returns
        -------
        list
            sub-vector genome
        """
        # todo add tests to make sure this logic works
        rnd_i = randint(0, self.size() - 1)
        end_i = randint(rnd_i, self.size() - 1)
        return self.vector[rnd_i:end_i + 1]

    def replace_vector_part_random(self, inserted_part):
        """
        Replace a given vector part in a random position

        Parameters
        -------
        inserted_part: list
            new vector part to be inserted

        Returns
        -------
        list
            previous vector part of this vector genome
        """
        index = randint(0, self.size() - len(inserted_part))  # select a random index
        end_i = index + len(inserted_part)
        replaced_part = self.vector[index:end_i]
        self.vector = self.vector[:index] + inserted_part + self.vector[end_i:]
        return replaced_part

    def replace_vector_part(self, inserted_part, start_index):
        """
        Replace a given vector part in a given position

        Parameters
        -------
        inserted_part: list
            new vector part to be inserted

        start_index: int
            starting position to insert the new vector part from

        Returns
        -------
        list
            previous vector part of this vector genome
        """
        end_i = start_index + len(inserted_part)
        replaced_part = self.vector[start_index:end_i]
        self.vector = self.vector[:start_index] + inserted_part + self.vector[end_i:]
        return replaced_part

    def get_vector_part(self, index, end_i):
        """
        Return vector part from `index` to `end_i`

        Parameters
        -------
        index: int
            starting index

        end_i: int
            end index

        Returns
        -------
        list
            sub-vector genome
        """
        return self.vector[index:end_i]

    def cell_value(self, index):
        """
        Get vector cell value in a given index

        Parameters
        -------
        index: int
            cell index

        Returns
        -------
        object
            vector cell value
        """
        return self.vector[index]

    def set_cell_value(self, index, value):
        """
        Set vector cell value in a given index

        Parameters
        -------
        index: int
            cell index

        value: object
            new cell value

        Returns
        -------
        None
        """
        self.vector[index] = value

    @abstractmethod
    def get_random_number_in_bounds(self, index):
        """
        Returns a random value in vector bounds

        Parameters
        -------
        index: int
            cell index

        Returns
        -------
        object
            vector cell value
        """
        raise ValueError("get_random_number is an abstract method in vector individual")

    def execute(self, *args, **kwargs):
        """
        Execute the vector.
        Input is a numpy array or keyword arguments (but not both).

        Parameters
        ----------
        args : arguments
            A numpy array, this is mostly relevant to GP representation.

        kwargs : keyword arguments
            Input to program, this is mostly relevant to GP representation.

        Returns
        -------
        object
            Vector (genome) of this individual.
        """
        return self.get_vector()

    def show(self):
        """
        Print out a simple textual representation of the vector.

        Returns
        -------
        None.
        """
        print(self.vector)

# end class Vector

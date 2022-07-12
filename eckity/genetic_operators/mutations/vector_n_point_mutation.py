from random import choices

from eckity.genetic_operators.failable_operator import FailableOperator


class VectorNPointMutation(FailableOperator):
    """
    Vector N Point Mutation.

    Randomly chooses N vector cells and performs a small change in their values.

    Parameters
    ----------
    n : int
        Number of mutation points.

    probability : float
        The probability of the mutation operator to be applied

    arity : int
        The number of individuals this mutation is applied on

    mut_val_getter: callable
        Returns a mutated value of a certain cell

    success_checker: callable
        Checks if a given (mutated) cell value is legal

    events: list of strings
        Events to publish before/after the mutation operator
    """
    def __init__(self, n=1, probability=1, arity=1, mut_val_getter=None,
                 success_checker=None, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        self.n = n

        if success_checker is None:
            success_checker = self.default_success_checker
        self.success_checker = success_checker

        if mut_val_getter is None:
            mut_val_getter = self.default_mut_val_getter
        self.mut_val_getter = mut_val_getter

    @staticmethod
    def default_mut_val_getter(vec, idx):
        """
        Default implementation for mutated value getter

        Parameters
        ----------
        vec : Vector
            a vector individual

        idx : int
            vector cell index

        Returns
        ----------
        object
            Mutated vector cell value
        """
        return vec.get_random_number_in_bounds(vec, idx)

    @staticmethod
    def default_success_checker(old_vec, new_vec):
        return new_vec.check_if_in_bounds()

    def attempt_operator(self, individuals, attempt_num):
        """
        Attempt to perform the mutation operator

        Parameters
        ----------
        individuals : list of individuals
            individuals to mutate

        attempt_num : int
            Current attempt number

        Returns
        ----------
        tuple of (bool, list of individuals)
            first return value determines if the the attempt succeeded
            second return value is the operator result
        """
        succeeded = True
        for individual in individuals:
            old_individual = individual.clone()

            # randomly select n points of the vector (without repetitions)
            m_points = choices(range(individual.size()), k=self.n)
            # obtain the mutated values
            mut_vals = [self.mut_val_getter(individual, m_point) for m_point in m_points]

            # update the mutated value in-place
            for m_point, mut_val in zip(m_points, mut_vals):
                individual.set_cell_value(m_point, mut_val)

            if not self.success_checker(old_individual, individual):
                succeeded = False
                break

        self.applied_individuals = individuals
        return succeeded, individuals

    def on_fail(self, payload):
        """
        The required fix when the operator fails, does nothing by default and can be overridden by subclasses

        Parameters
        ----------
        payload : object
            relevant data for on_fail (usually the individuals that the mutation was attempted to be applied on)
        """
        pass

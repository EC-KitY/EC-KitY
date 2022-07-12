from eckity.genetic_operators.genetic_operator import GeneticOperator
from abc import abstractmethod


class FailableOperator(GeneticOperator):
    """
    Genetic operator that has a chance of failing.

    For example, adding a gaussian noise to a FloatVector cell might exceed the legal bounds of the vector.
    In that case, the Gauss Mutation fails.

    Parameters
    -------
    probability: float
        the probability of the operator to be applied

    arity: int
        number of individuals to be applied on

    events: list of strings
        events to be published before, after and during the operator

    attempts: int
        number of attempts to be made during the operator execution
    """
    def __init__(self, probability=0.05, arity=0, events=None, attempts=5):
        super().__init__(probability, arity, events)
        if attempts <= 1:
            raise ValueError("Number of attempts must be at least 1")
        self.attempts = attempts

    # TODO add event of on fail or on fail all retries
    def apply(self, payload):
        """
        Apply the operator, with a chance of failing.

        Attempt to apply the operator `attempts` times, finish by succeeding in one of the attempts or by failing
        all attempts and executing `on_fail` method.

        Parameters
        -------
        payload: object
            relevant data for the applied operator (usually a list of individuals)
        """
        for i in range(self.attempts):
            # attempt to execute the operator
            succeeded, result = self.attempt_operator(payload, i)

            # return if succeeded
            if succeeded:
                return result
        # after all attempts failed, execute the `on_fail` mechanism
        return self.on_fail(payload)

    @abstractmethod
    def attempt_operator(self, payload, attempt_num):
        """
        A single attempt of the operator

        Parameters
        -------
        payload: object
            relevant data for the applied operator (usually a list of individuals)

        attempt_num: int
            current attempt number

        Returns
        -------
        (bool, object)
            tuple of (succeeded or not, result value)
        """
        pass

    @abstractmethod
    def on_fail(self, payload):
        """
        What to do when all operator attempts failed

        This method is called once all operator attempts have failed

        Parameters
        -------
        payload: object
            relevant data for the failure handling mechanism (usually a list of individuals)

        Returns
        -------
        object
            result value
        """
        pass

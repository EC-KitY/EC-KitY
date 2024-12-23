"""
This module implements the Operator class
"""

from abc import ABC, abstractmethod
from typing import Any

from eckity.before_after_publisher import BeforeAfterPublisher


class Operator(BeforeAfterPublisher, ABC):
    def __init__(self, arity=1, events=None, event_names=None):
        super().__init__(events=events, event_names=event_names)
        self.applied_individuals = None
        self.arity = arity

    @abstractmethod
    def apply_operator(self, payload):
        pass

    def initialize(self):
        pass

    def act(self, payload=None):
        """
        Applies the subclass-specific operator on the given payload,
        and publishing events before and after the operator execution

        Parameters
        ----------
        payload:
            operands to apply the operator on

        Returns
        -------
        the return value of the operator implemented in the sub-class
        """
        self._assert_arity(payload)
        return self.act_and_publish_before_after(
            lambda: self.apply_operator(payload)
        )

    def _assert_arity(self, payload: Any) -> None:
        if not isinstance(payload, list):
            if self.arity != 1:
                raise ValueError(
                    f"Received payload of type {type(payload)}, "
                    f"expected arity of 1 but got {self.arity}."
                )

        elif (n_individuals := len(payload)) != self.arity:
            raise ValueError(
                f"Expected {self.arity} individuals,"
                f"but received {n_individuals}."
            )

    def get_operator_arity(self):
        """
        Getter method for the number of operands this operator is applied on
        For example, a crossover that exchanges subtrees of 2 individuals will have an arity of 2

        Returns
        -------
        int
            number of operands this operator is applied on
        """
        return self.arity

    def event_name_to_data(self, event_name):
        if event_name == "after_operator":
            return {"applied_individuals": self.applied_individuals}
        if event_name == "before_operator":
            return {}
        else:
            return {}

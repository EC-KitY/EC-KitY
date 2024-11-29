from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from overrides import override

from eckity.base.utils import arity

from .utils import get_func_types


class TreeNode(ABC):
    """
    GP Node

    Attributes
    ----------
    node_type : type
        node type
    """

    def __init__(self, node_type: Optional[type] = None) -> None:
        self.node_type = node_type

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __eq__(self, other):
        return (
            isinstance(other, TreeNode) and self.node_type is other.node_type
        )

    def __repr__(self):
        return str(self)


class FunctionNode(TreeNode):
    def __init__(
        self,
        function: Callable,
    ) -> None:
        # infer the return type of the function
        func_types = get_func_types(function)
        return_type = func_types[-1] if func_types else None
        self.n_args = arity(function)

        if 0 < len(func_types) < self.n_args + 1:
            raise ValueError(
                f"Function {function.__name__} has missing type hints."
                f"Please provide type hints for all arguments and return type."
            )

        super().__init__(return_type)

        self.function = function

    @override
    def __eq__(self, other: object) -> bool:
        """
        Compare two FunctionNodes for equality.
        Function nodes are equal if they have the same function.

        Parameters
        ----------
        other : object
            The object to compare to

        Returns
        -------
        bool
            True if the two FunctionNodes are equal, False otherwise
        """
        return (
            super().__eq__(other)
            and isinstance(other, FunctionNode)
            and self.function == other.function
        )

    @override
    def __str__(self) -> str:
        return self.function.__name__


class TerminalNode(TreeNode):
    def __init__(
        self,
        value: Any,
        node_type: Optional[type] = None
    ) -> None:
        super().__init__(node_type)
        self.value = value

    @override
    def __eq__(self, other):
        return (
            super().__eq__(other)
            and isinstance(other, TerminalNode)
            and self.value is other.value
        )

    def __str__(self):
        return str(self.value)

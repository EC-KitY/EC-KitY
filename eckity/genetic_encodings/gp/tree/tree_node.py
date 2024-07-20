from abc import ABC, abstractmethod
from numbers import Number
from types import NoneType
from typing import Any, Callable, Dict, List, get_type_hints

import numpy as np
from overrides import override

from eckity.base.utils import arity


class TreeNode(ABC):
    """
    GP Node

    Attributes
    ----------
    node_type : type
        node type
    """

    def __init__(self, node_type: type = NoneType) -> None:
        self.node_type = node_type

    @abstractmethod
    def execute(self, **kwargs):
        """
        Returns the value of this node
        Return type must match the type of this node
        """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    def __eq__(self, other):
        return (
            isinstance(other, TreeNode) and self.node_type == other.node_type
        )

    def __repr__(self):
        return str(self)


class FunctionNode(TreeNode):
    def __init__(
        self,
        function: Callable,
    ) -> None:
        # infer the return type of the function
        func_types = FunctionNode.get_func_types(function)
        return_type = func_types[-1] if func_types else NoneType
        self.n_args = arity(function)

        if 0 < len(func_types) < self.n_args + 1:
            raise ValueError(
                f"Function {function.__name__} has missing type hints."
                f"Please provide type hints for all arguments and return type."
            )

        super().__init__(return_type)

        self.function = function

    @override
    def execute(self, **kwargs):
        """
        Recursively execute the tree by traversing it in a depth-first order
        """
        kw_types: Dict[str, type] = get_type_hints(self.function)

        # assert that the types of the arguments match the expected types
        if kw_types:
            for k, v in kwargs.items():
                if kw_types[k] is not NoneType and kw_types[k] is not type(v):
                    raise TypeError(
                        f"Expected {k} to be of type {kw_types[k]}, "
                        f"got {type(v)}."
                    )

        return self.function(**kwargs)

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

    @staticmethod
    def get_func_types(f: Callable) -> List[type]:
        """
        Return list of function types in the following format:
        [type_arg_1, type_arg_2, ..., type_arg_n, return_type]

        Parameters
        ----------
        f : Callable
            function (builtin or user-defined)

        Returns
        -------
        List[type]
            List of function types, sorted by argument order
            with the return type as the last element.
            For untyped functions, NoneType is used.

        Examples
        --------
        >>> def f(x: int, y: float) -> float:
        ...     return x + y
        >>> FunctionNode.get_func_types(f)
        [int, float, float]

        >>> def f(x, y):
        ...     return x + y
        >>> FunctionNode.get_func_types(f)
        [NoneType, NoneType, NoneType]
        """
        params_types: Dict = get_type_hints(f)
        type_list = list(params_types.values())
        if not type_list:
            # If we don't have type hints, assign None types
            type_list = [NoneType] * (arity(f) + 1)
        return type_list


class TerminalNode(TreeNode):
    def __init__(self, value: Any, node_type=NoneType, parent=None) -> None:
        super().__init__(node_type)
        self.value = value

    @override
    def execute(self, **kwargs):
        """Recursively execute the tree by traversing it in a depth-first order"""

        if isinstance(self.value, Number):  # terminal is a constant
            return self.value

        # terminal is a variable, return its value if type matches
        kwarg_val = kwargs[self.value]

        # kwarg might be a numpy array
        kwarg_type = (
            type(kwarg_val.item(0))
            if isinstance(kwarg_val, np.ndarray)
            else type(kwarg_val)
        )

        if self.node_type is not NoneType and self.node_type != kwarg_type:
            raise TypeError(
                f"Expected {self.value} to be of type {self.node_type},"
                f"got {kwarg_type}."
            )
        return kwarg_val

    @override
    def __eq__(self, other):
        return (
            super().__eq__(other)
            and isinstance(other, TerminalNode)
            and self.value == other.value
        )

    def __str__(self):
        return str(self.value)

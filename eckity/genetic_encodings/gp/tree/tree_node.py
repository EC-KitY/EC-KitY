from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from eckity.base.utils import arity

from overrides import override


class TreeNode(ABC):
    """
    GP Node

    Attributes
    ----------
    node_type : type
        node type
    """

    def __init__(self, node_type: Optional[type] = None) -> None:
        self.node_type: type = node_type

    @abstractmethod
    def apply(self):
        """
        Returns the value of this node
        Return type must match the type of this node
        """
        pass


class FunctionNode(TreeNode):
    def __init__(self, function: Callable) -> None:
        # infer the return type of the function
        func_types = FunctionNode.get_func_types(function)
        return_type = func_types[-1] if func_types else None

        super().__init__(return_type)
        self.function = function
        self.children: List[TreeNode] = []

    @override
    def apply(self):
        return self.function(*[child.apply() for child in self.children])

    def add_child(self, child: TreeNode) -> None:

        child_idx = len(self.children)

        # Check if there are too many children
        if child_idx >= arity(self.function):
            raise ValueError(
                f"Too many children for function {self.function}."
            )

        # Check if child is of the correct type
        func_types = FunctionNode.get_func_types(self.function)
        if func_types:
            # Check if the child is of the correct type
            expected_type = func_types[child_idx]
            if child.node_type != expected_type:
                raise TypeError(
                    f"Expected Child {child_idx} of function {self.function} "
                    f"to be of type {expected_type}. Got {child.node_type}"
                )
        self.children.append(child)

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
            with the return type as the last element
        """
        params_types: Dict = get_type_hints(f)
        return list(params_types.values())


class TerminalNode(TreeNode):
    def __init__(self, value: Any, node_type=None) -> None:
        # set node type, not always using type inference since
        # value may be a string variable with int value
        if node_type is None:
            node_type = type(value)
        super().__init__(node_type)
        self.value = value

    @override
    def apply(self):
        return self.value

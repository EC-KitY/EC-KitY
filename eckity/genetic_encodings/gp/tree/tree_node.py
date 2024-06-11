from abc import ABC, abstractmethod
from types import BuiltinFunctionType, FunctionType
from typing import Any, Dict, List, Union, get_type_hints

from overrides import override


class TreeNode(ABC):
    """
    GP Node

    Attributes
    ----------
    node_type : type
        node type
    value : node_type
        node value
    children: List[GPNode]
        list of children nodes
    """

    def __init__(self, node_type) -> None:
        self.node_type: type = node_type

    @abstractmethod
    def apply(self):
        """
        Returns the value of this node
        Return type must match the type of this node
        """
        pass


class FunctionNode(TreeNode):
    def __init__(
        self, function: Union[FunctionType, BuiltinFunctionType]
    ) -> None:
        super().__init__(type(function))
        self.function = function
        self.children: List[TreeNode] = []

    @override
    def apply(self):
        return self.function(*[child.apply() for child in self.children])

    def add_child(self, child: TreeNode):
        # Check if child is of the correct type
        params_types: Dict = get_type_hints(self.function)
        func_types = list(params_types.values())
        child_idx = len(self.children)

        # Check if there are too many children
        if child_idx >= len(func_types):
            raise ValueError(
                f"Too many children for function {self.function}."
            )

        # Check if the child is of the correct type
        expected_type = func_types[child_idx]
        if child.node_type != expected_type:
            raise ValueError(
                f"Expected Child {child_idx} of function {self.function} "
                f"to be of type {expected_type}. Got {child.node_type}"
            )
        self.children.append(child)


class TerminalNode(TreeNode):
    def __init__(self, value: Any) -> None:
        super().__init__(type(value))
        self.value = value

    @override
    def apply(self):
        return self.value

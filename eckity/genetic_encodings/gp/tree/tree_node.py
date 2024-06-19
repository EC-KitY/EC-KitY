from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, get_type_hints
from eckity.base.utils import arity
from numbers import Number


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
    def execute(self, **kwargs):
        """
        Returns the value of this node
        Return type must match the type of this node
        """
        # TODO: Add type checking
        pass

    @abstractmethod
    def depth(self, d=0):
        """Recursively compute depth"""
        pass

    @abstractmethod
    def generate_tree_code(self, prefix, result):
        """Recursively produce a simple textual printout of the tree"""
        pass

    def __eq__(self, other):
        return (
            isinstance(other, TreeNode) and self.node_type == other.node_type
        )

    def replace_child(self, old_child, new_child):
        pass

    def size(self):
        return 1

    def filter_by_type(self, node_type, nodes):
        if self.node_type == node_type:
            nodes.append(self)
        return nodes


class FunctionNode(TreeNode):
    def __init__(
        self, function: Callable, children: List[TreeNode] = None
    ) -> None:
        # infer the return type of the function
        func_types = FunctionNode.get_func_types(function)
        return_type = func_types[-1] if func_types else None

        if 0 < len(func_types) < arity(function) + 1:
            raise ValueError(
                f"Function {function.__name__} has missing type hints."
                f"Please provide type hints for all arguments and return type."
            )

        super().__init__(return_type)
        self.function = function
        self.children: List[TreeNode] = children if children else []

    @override
    def execute(self, **kwargs):
        """
        Recursively execute the tree by traversing it in a depth-first order
        """

        arglist = []
        for child in self.children:
            res = child.execute(**kwargs)
            arglist.append(res)
        return self.function(*arglist)

    def add_child(self, child: TreeNode) -> None:

        child_idx = len(self.children)

        # Check if there are too many children
        if child_idx >= arity(self.function):
            raise ValueError(
                f"Too many children for function {self.function}."
            )

        # Check if child is of the correct type
        func_types = FunctionNode.get_func_types(self.function)

        if not func_types:
            # If we don't have type hints, assign None types
            func_types = [None] * (arity(self.function) + 1)

        # Check if the child is of the correct type
        expected_type = func_types[child_idx]
        if child.node_type != expected_type:
            raise TypeError(
                f"Expected Child {child_idx} of function "
                f"{self.function.__name__} to be {expected_type}. "
                f"Got {child.node_type}."
            )

        self.children.append(child)

    @override
    def depth(self, d=0):
        """Recursively compute depth"""
        return 1 + max([child.depth(d) for child in self.children], default=0)

    @override
    def size(self):
        return 1 + sum([child.size() for child in self.children])

    @override
    def generate_tree_code(self, prefix, result):
        """Recursively produce a simple textual printout of the tree"""
        result.append(f'{prefix}{self.function.__name__}{"("}\n')
        for i, child in enumerate(self.children):
            child.str_rec(prefix + "   ", result)
            result.append(",")
            if i < len(self.children) - 1:
                result.append("\n")
        result.append(prefix + ")")

    @override
    def filter_by_type(self, node_type, nodes):
        nodes = super().filter_by_type(node_type, nodes)
        for child in self.children:
            nodes = child.filter_by_type(node_type, nodes)
        return nodes

    @override
    def replace_child(self, old_child, new_child):
        for i, child in enumerate(self.children):
            if child == old_child:
                self.children[i] = new_child
                return
            child.replace_child(old_child, new_child)

    @override
    def __eq__(self, other):
        return (
            super().__eq__(other)
            and isinstance(other, FunctionNode)
            and self.function == other.function
            and self.children == other.children
        )

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
        super().__init__(node_type)
        self.value = value

    @override
    def depth(self, d=0):
        """Recursively compute depth"""
        return 1

    @override
    def execute(self, **kwargs):
        """Recursively execute the tree by traversing it in a depth-first order
        (pos is a size-1 list so as to pass "by reference" on successive recursive calls).
        """

        if isinstance(self.value, Number):  # terminal is a constant
            return self.value
        else:  # terminal is a variable, return its value
            return kwargs[self.value]

    @override
    def generate_tree_code(self, prefix, result):
        """Recursively produce a simple textual printout of the tree"""
        result.append(f"{prefix}{str(self.value)}")

    @override
    def __eq__(self, other):
        return (
            super().__eq__(other)
            and isinstance(other, TerminalNode)
            and self.value == other.value
        )

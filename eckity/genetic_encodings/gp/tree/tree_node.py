from typing import List, get_type_hints


class TreeNode:
    """
    GP Node

    Attributes
    ----------
    node_type : type_
        node type
    value : _type_
        node value
    children: List[GPNode]
        list of children nodes
    """

    def __init__(self, node_type, value) -> None:
        self.children: List[TreeNode] = []
        self.node_type: type = None
        self.value: self.node_type = None

        if self.node_type is None:
            self.node_type = type(self.value)

    def apply(self):
        if isinstance(self.value, callable):
            return self.value(*[child.apply() for child in self.children])
        else:
            return self.value

    def add_child(self, child: "TreeNode"):
        if type(self.value) is not callable:
            raise ValueError("Cannot add child to terminal node.")

        # Check if child is of the correct type
        func_types = list(get_type_hints(self.value).values())
        child_idx = len(self.children)

        # Check if there are too many children
        if child_idx >= len(func_types):
            raise ValueError(f"Too many children for function {self.value}.")

        # Check if the child is of the correct type
        if not isinstance(child.value, func_types[child_idx]):
            raise ValueError(
                f"Child {child_idx} of function {self.value} "
                f"should be of type {func_types[child_idx]}."
            )
        self.children.append(child)

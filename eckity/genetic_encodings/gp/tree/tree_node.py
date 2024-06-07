from typing import List


class TreeNode:
    """
    GP Node

    Attributes
    ----------
    node_type : type_
        node type
    value : _type_
        node value
    children_types: List[type]
        function argument types
        for instance, f_add GPNode should have children_types = [int, int]
    children: List[GPNode]
        list of children nodes
    """

    def __init__(self, node_type, value) -> None:
        self.children: List[TreeNode] = []
        self.node_type: type = None
        self.children_types: List[type] = []  # function argument types
        self.value: self.node_type = None

        if self.node_type is None:
            self.node_type = type(self.value)

    def apply(self):
        if isinstance(self.value, callable):
            return self.value(*[child.apply() for child in self.children])
        else:
            return self.value

    def add_child(self, child: "TreeNode"):
        self.children.append(child)

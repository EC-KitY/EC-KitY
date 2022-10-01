class TreeNode:
    def __init__(self, num_of_descendants=0, type=None):
        """
        This class implements a base tree node.
        Parameters
        ----------
        num_of_descendants : int
            Number of node's children nodes. The default is 0.
        type : variable's type
            Terminal - its type.
            Function - its return value type.
            Not typed value is private case on a value. The default is None.
        """
        self.num_of_descendants = num_of_descendants
        self.type = type


class FunctionNode(TreeNode):
    def __init__(self, function=None, num_of_parameters=0, parameters=None, type=None):
        """
        This class implements a function tree node.
        Parameters
        ----------
        function : numpy function
            The function used as internal nodes in the GP tree. The default is None.
        num_of_parameters : int
            Number parameters the function receives. The default is 0.
        Relevant only for typed_gp:
        --------------------------
        parameters : list of types
            The list of types of parameters the function receives. The default is None.
        type : variable's type
            Function - its return value type.
            Not typed value is private case on a value. The default is None.
        """
        super().__init__(num_of_parameters, type)
        self.function = function
        self.parameters = parameters


class TerminalNode(TreeNode):
    def __init__(self, value=None, type=None):
        """
        This class implements a terminal tree node.
        Parameters
        ----------
        value : any
            The value of the terminal used in the GP-tree leave. The default is none.
        type : variable's type
            Terminal - its type.
            Not typed value is private case on a value. The default is None.
        """
        super().__init__(0, type)
        self.value = value


class RootNode(FunctionNode):
    def __init__(self, function=None, num_of_parameters=1, parameters=None, type=None):
        """
        This class implements a Root tree node.
         Parameters
        ----------
        function : numpy function
            The function used as internal nodes in the GP tree. The default is None.
        num_of_parameters : int
            Number parameters the function receives. The default is 0.
        Relevant only for typed_gp:
        --------------------------
        parameters : list of types
            The list of types of parameters the function receives. The default is None.
        type : variable's type
            Function - its return value type.
            Not typed value is private case on a value. The default is None.
        """
        super().__init__(function, num_of_parameters, parameters, type)

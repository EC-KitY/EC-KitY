"""
This module implements the tree class.
"""

import logging
import random
from numbers import Number
from types import NoneType
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    get_type_hints,
)

import numpy as np

from eckity.base.untyped_functions import f_add, f_div, f_mul, f_sub
from eckity.fitness import Fitness, GPFitness
from eckity.genetic_encodings.gp.tree.tree_node import (
    FunctionNode,
    TerminalNode,
    TreeNode,
)
from eckity.individual import Individual

from .utils import generate_args

logger = logging.getLogger(__name__)


class Tree(Individual):
    """
    A tree optimized for genetic programming operations.
    It is represented by a list of nodes in depth-first order.
    There are two types of nodes: functions and terminals.

    (tree is not meant as a stand-alone,
    parameters are supplied through the call from the Tree Creators)

    Parameters
    ----------
    function_set: list, default=None
        List of functions used as internal nodes in the GP tree.

    terminal_set: Dict[Any, type], default=None
        Mapping of terminal nodes and their types.
        In the untyped case, all types are NoneType.
    """

    def __init__(
        self,
        fitness: Fitness = GPFitness(),
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        tree: List[TreeNode] = None,
        erc_range: Optional[Union[Tuple[float, float], Tuple[int, int]]] = (
            -1.0,
            1.0,
        ),
    ):
        """
        GP Tree Individual.

        Parameters
        ----------
        fitness : Fitness
            Manages fitness state, by default GPFitness
        function_set: List[Callable], default=None
            List of functions used as internal nodes in the GP tree.
        terminal_set : Union[Dict[Any, type], List[Any]], optional
            Mapping of terminal nodes and their types.
            In the untyped case, all types are NoneType.
            Lists are treated as untyped, and will be assigned NoneType.
        tree : List[TreeNode], optional
            Actual tree representation, by default None
        erc_range : tuple of float or int, optional
            Range of Ephemeral random constant values, by default (-1.0, 1.0)

        Raises
        ------
        ValueError
            If typed function is used with untyped terminals.
        """
        super().__init__(fitness)
        if function_set is None:
            function_set = [f_add, f_sub, f_mul, f_div]

        if terminal_set is None:
            terminal_set = {"x": float, "y": float, "z": float}

        # untyped case - convert to dictionary of NoneTypes.
        if isinstance(terminal_set, list):
            # check if any function has type hints
            if any(f.__annotations__ for f in function_set):
                raise ValueError(
                    "Detected typed function with untyped terminal set. \
                        Please provide a dictionary with types for terminals."
                )

            terminal_set = {t: NoneType for t in terminal_set}

        self.function_set = function_set
        self.terminal_set = terminal_set

        self.erc_range = erc_range

        if tree is None:
            tree = []
        self.tree = tree  # actual tree representation

    def size(self) -> int:
        """
        Compute size of tree.

        Returns
        -------
        int
            tree size (= number of nodes).
        """
        return len(self.tree)

    def add_tree(self, node: TreeNode) -> None:
        self.tree.append(node)

    def empty_tree(self) -> None:
        self.tree = []

    def depth(self):
        """
        Compute depth of tree (maximal path length to a leaf).

        Returns
        -------
        int
            tree depth.
        """
        return self._depth([0], d=0)

    def _depth(self, pos, d):
        """Recursively compute depth
        (pos is a size-1 list so as to pass "by reference"
        on successive recursive calls).
        """

        node = self.tree[pos[0]]

        depths = []
        if isinstance(node, FunctionNode):
            for _ in range(node.n_args):
                pos[0] += 1
                depths.append(1 + self._depth(pos, d + 1))
            return max(depths)
        else:
            return 0

    def random_function(self, node_type=NoneType) -> Optional[FunctionNode]:
        functions_types = {
            func: get_type_hints(func).get("return", NoneType)
            for func in self.function_set
        }
        relevant_functions = [
            func
            for func in self.function_set
            if functions_types[func] == node_type
        ]

        # Return None in case there are no functions of the given type
        if not relevant_functions:
            return None

        func = random.choice(relevant_functions)
        return FunctionNode(func)

    def random_terminal(self, node_type=NoneType) -> Optional[TerminalNode]:
        """Select a random terminal, including constants from ERC range"""
        relevant_terminals = [
            term
            for term, term_type in self.terminal_set.items()
            if term_type == node_type
        ]

        if self.erc_range is not None and (
            node_type is NoneType or issubclass(node_type, Number)
        ):
            relevant_terminals.append(
                random.uniform(*self.erc_range)
                if type(self.erc_range[0]) is float
                else random.randint(*self.erc_range)
            )

        # Return None in case there are no terminals of the given type
        if not relevant_terminals:
            return None

        terminal = random.choice(relevant_terminals)

        # get the type of the terminal (erc terminal will be a float or int)
        node_type = self.terminal_set.get(terminal, type(terminal))
        
        return TerminalNode(terminal, node_type=node_type)

    def execute(self, *args, **kwargs) -> object:
        """
        Execute the program (tree).
        Input is a numpy array or keyword arguments (but not both).

        Parameters
        ----------
        args : arguments
            A numpy array.

        kwargs : keyword arguments
            Input to program, including every variable in the terminal set as a keyword argument.
            For example, if `terminal_set=['x', 'y', 'z', 0, 1, -1]`
            then call `execute(x=..., y=..., z=...)`.

        Returns
        -------
        object
            Result of tree execution.
        """

        if not self.tree:
            raise ValueError("Tree is empty, cannot execute.")

        reshape = False
        if args:  # numpy array -- convert to kwargs
            try:
                X = args[0]
                kwargs = generate_args(X)
                reshape = True
            except Exception:
                raise ValueError(
                    f"Bad argument to tree.execute, must be numpy array or kwargs: {args}"
                )

        kw = list(kwargs.keys())

        bad_vars = [item for item in kw if item not in self.terminal_set]
        if len(bad_vars) > 0:
            raise ValueError(
                f"received variable arguments not in terminal set: {bad_vars}"
            )

        missing_vars = [item for item in self.terminal_set if item not in kw]
        if len(missing_vars) > 0:
            raise ValueError(
                f"Missing variable terminals as execute kwargs: {missing_vars}"
            )

        res = self._execute([0], **kwargs)

        if reshape and (isinstance(res, Number) or res.shape == np.shape(0)):
            # sometimes a tree degenrates to a scalar value
            res = np.full_like(X[:, 0], res)
        return res

    def _execute(self, pos, **kwargs):
        """
        Recursively execute the tree by traversing it in a depth-first order
        (pos is a size-1 list so as to pass "by reference"
        on successive recursive calls).
        """

        node = self.tree[pos[0]]

        if isinstance(node, FunctionNode):
            arglist = []
            for _ in range(node.n_args):
                pos[0] += 1
                res = self._execute(pos, **kwargs)
                arglist.append(res)
            return node(*arglist)
        else:  # terminal
            if isinstance(node, Number):  # terminal is a constant
                return node
            else:  # terminal is a variable, return its value
                return kwargs[node]

    def filter_tree(self, filter_func: Callable) -> None:
        return [node for node in self.tree if filter_func(node)]

    def get_random_numeric_node(self) -> Optional[TerminalNode]:
        """
        Get a random leaf node of the tree.

        Parameters
        ----------
        node_type : type, default=None
            Type of node to return.

        Returns
        -------
        TreeNode
            Random leaf node.
        """
        erc_nodes = []
        self.filter_nodes(
            lambda node: isinstance(node, TerminalNode)
            and isinstance(node.value, Number),
            erc_nodes,
        )
        return random.choice(erc_nodes) if erc_nodes else None

    def random_subtree(self, node_type=NoneType) -> Optional[TreeNode]:
        relevant_nodes = self.filter_tree(
            lambda node: node.node_type in [Any, NoneType]
            or node.node_type == node_type
        )
        return random.choice(relevant_nodes) if relevant_nodes else None

    def replace_subtree(
        self, old_subtree_root: TreeNode, new_subtree: List[TreeNode]
    ) -> None:
        """
        Replace the subtree starting at `index` with `subtree`

        Parameters
        ----------
        subtree - new subtree to replace the existing subtree in the tree

        Returns
        -------
        None
        """
        index = self.tree.index(old_subtree_root)
        end_i = self._find_subtree_end([index])
        left_part = self.tree[:index]
        right_part = self.tree[(end_i + 1) :]
        self.tree = left_part + new_subtree + right_part

    def _find_subtree_end(self, pos):
        """find index of final node of subtree that starts at `pos`
        (pos is a size-1 list so as to pass "by reference" on successive recursive calls).
        """

        node = self.tree[pos[0]]

        if node in self.function_set:
            for _ in range(self.arity[node]):
                pos[0] += 1
                self._find_subtree_end(pos)

        return pos[0]

    def _str_rec(self, prefix, pos, result):
        """Recursively produce a simple textual printout of the tree
        (pos is a size-1 list so as to pass "by reference" on successive recursive calls).
        """

        node = self.tree[pos[0]]
        if isinstance(node, FunctionNode):
            result.append(f"{prefix}{str(node)}(\n")
            for i in range(node.n_args):
                pos[0] += 1
                self._str_rec(prefix + "\t", pos, result)
                result.append(",")
                if i < node.n_args - 1:
                    result.append("\n")
            result.append(prefix + ")")
        else:  # terminal
            result.append(f"{prefix}{str(node)}")

    def __str__(self):
        """
        Return a simple textual representation of the tree.

        Returns
        -------
        str
            String representation of the tree.

        Examples
        --------
        untyped case:
        >>> ut = Tree(terminal_set=[x, y, z], function_set=[f_add])
        >>> ut.tree = [FunctionNode(f_add), TerminalNode(1), TerminalNode(2)]
        >>> print(tree)
        def func_0(x, y, z):
            return f_add(x, y)

        typed case:
        >>> tt = Tree(terminal_set={x: float, y: float},
        ...           function_set=[typed_add])
        >>> tt.tree = [FunctionNode(typed_add),
        ...            TerminalNode(1.0, float),
        ...            TerminalNode(2.0, float)]
        >>> print(tree)
        def func_0(x: float, y: float) -> float:
            return typed_add(1.0, 2.0)
        """
        args = (
            list(self.terminal_set.keys())
            if NoneType in self.terminal_set.values()
            else [f"{k}: {v}" for k, v in self.terminal_set.items()]
        )
        result = [f"def func_{self.id}({', '.join(args)}):\n\treturn "]
        self._str_rec("\t", [0], result)
        return "".join(result)

    def show(self):
        """
        Print out a simple textual representation of the tree.

        Returns
        -------
        None.
        """
        logger.info("\n" + str(self))

    def __repr__(self):
        return str(self)


# end class tree

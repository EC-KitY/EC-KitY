"""
This module implements the tree class.
"""

import logging
import random
from numbers import Number
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from eckity.fitness import Fitness, GPFitness
from eckity.genetic_encodings.gp.tree.tree_node import (
    FunctionNode,
    TerminalNode,
    TreeNode,
)
from eckity.individual import Individual

from .utils import generate_args, get_func_types, get_return_type

logger = logging.getLogger(__name__)


class Tree(Individual):
    """
    A tree optimized for genetic programming operations.
    It is represented by a list of nodes in depth-first order.
    There are two types of nodes: functions and terminals.

    (tree is not meant as a stand-alone,
    parameters are supplied through the call from the Tree Creators)
    """

    def __init__(
        self,
        fitness: Fitness = GPFitness(),
        function_set: List[Callable] = None,
        terminal_set: Union[Dict[Any, type], List[Any]] = None,
        tree: List[TreeNode] = None,
        erc_range: Optional[
            Union[Tuple[float, float], Tuple[int, int]]
        ] = None,
        root_type: Optional[type] = None,
        update_parents: bool = False
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
            In the untyped case, all types are None.
            Lists are treated as untyped, and will be assigned None.
        tree : List[TreeNode], optional
            Actual tree representation, by default None
        erc_range : tuple of float or int, optional
            Range of Ephemeral random constant values, by default None
        root_type: type, optional
            Root node type, by default None

        Raises
        ------
        ValueError
            If typed function is used with untyped terminals.
        """
        super().__init__(fitness, update_parents=update_parents)

        self.erc_range = erc_range

        function_set, terminal_set = self._handle_input_types(
            function_set, terminal_set, root_type
        )

        self.function_set = function_set
        self.terminal_set = terminal_set

        # actual tree representation
        if tree is None:
            tree = []
        self.tree = tree

        # this is the type of the execution result of the program (tree)
        self.root_type = root_type

    @property
    def root(self) -> TreeNode:
        return self.tree[0]

    @property
    def erc_type(self) -> Optional[Union[int, float]]:
        return type(self.erc_range[0]) if self.erc_range else None

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
        """Add a node to the tree following the defined type constrains"""
        if (
            self.size() == 0 and node.node_type == self.root_type
        ) or self._should_add([0], node):
            self.tree.append(node)
        else:
            raise ValueError(f"Could not add node {node} to tree {self.tree}")

    def _should_add(self, pos: int, new_node: TreeNode) -> bool:
        """
        Recursively find the parent function of the new node and check it
        matches by type to the expected parameter. (pos is a size-1 list
        so as to pass "by reference" on successive recursive calls).
        """
        if (
            pos[0] == self.size()
        ):  # reached the place the new node should be placed
            return True

        node = self.tree[pos[0]]
        res = None
        if isinstance(node, FunctionNode):
            func_types = get_func_types(node.function)
            for i in range(node.n_args):
                pos[0] += 1
                res = self._should_add(pos, node)
                if res:
                    return func_types[i] == new_node.node_type
                elif res is not None:
                    return res
        return res

    def empty_tree(self) -> None:
        self.tree = []

    def depth(self) -> int:
        """
        Compute depth of tree (maximal path length to a leaf).

        Returns
        -------
        int
            tree depth.
        """
        return self._depth([0], d=0)

    def _depth(self, pos, d) -> int:
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

    def random_function(
        self,
        node_type: Optional[type] = None
    ) -> Optional[FunctionNode]:
        functions_types = {
            func: get_return_type(func) for func in self.function_set
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

    def random_terminal(
        self,
        node_type: Optional[type] = None
    ) -> Optional[TerminalNode]:
        """Select a random terminal, including constants from ERC range"""
        relevant_terminals = [
            term
            for term, term_type in self.terminal_set.items()
            if term_type == node_type
        ]

        if self.erc_range is not None and (
            node_type is None or issubclass(node_type, Number)
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

        # erc terminal will be typeless in untyped case,
        # and int/float in typed case
        default_type = None if node_type is None else type(terminal)
        node_type = self.terminal_set.get(terminal, default_type)

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
            Input to program, including every variable
            in the terminal set as a keyword argument.
            For example, if `terminal_set=['x', 'y', 'z']`
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
                    f"Bad argument to tree.execute, "
                    f"must be np array or kwargs: {args}"
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
            return node.function(*arglist)
        else:  # terminal
            if node.value in kwargs:
                # terminal is a variable, return its value
                return kwargs[node.value]
            else:  # terminal is a constant
                return node.value

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
        erc_nodes = self.filter_tree(
            lambda node: isinstance(node, TerminalNode)
            and isinstance(node.value, Number)
        )
        return random.choice(erc_nodes) if erc_nodes else None

    def random_subtree(
        self,
        node_type: Optional[type] = None
    ) -> Optional[List[TreeNode]]:
        relevant_nodes = self.filter_tree(
            lambda node: node.node_type is None  # untyped case
            or (
                node_type is None and node != self.root
            )  # typed case with first invocation
            or (
                node.node_type == node_type and node != self.root
            )  # typed case with subsequent invocations
        )
        if not relevant_nodes:
            return None
        subtree_root = random.choice(relevant_nodes)

        return self._get_subtree_by_root(subtree_root)

    def _get_subtree_by_root(self, subtree_root: TreeNode) -> List[TreeNode]:
        start_i = self.tree.index(subtree_root)

        if start_i == 0:
            return self.tree

        end_i = self._find_subtree_end([start_i])
        return self.tree[start_i: end_i + 1]

    def replace_subtree(
        self, old_subtree: List[TreeNode], new_subtree: List[TreeNode]
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
        start_i = self.tree.index(old_subtree[0])
        end_i = start_i + len(old_subtree)
        left_part = self.tree[:start_i]
        right_part = self.tree[end_i:]
        self.tree = left_part + new_subtree + right_part

    def _find_subtree_end(self, pos):
        """find index of final node of subtree that starts at `pos`
        (pos is a size-1 list so as to pass "by reference" on successive recursive calls).
        """

        node = self.tree[pos[0]]

        if (
            isinstance(node, FunctionNode)
            and node.function in self.function_set
        ):
            for _ in range(node.n_args):
                pos[0] += 1
                self._find_subtree_end(pos)

        return pos[0]

    def _handle_input_types(
        self,
        function_set: List[Callable],
        terminal_set: Union[Dict[Any, type], List[str]],
        root_type: type
    ):
        if function_set is None:
            raise ValueError("Function set must be provided.")

        if terminal_set is None:
            raise ValueError("Terminal set must be provided.")

        for t in function_set:
            if not isinstance(t, Callable):
                raise ValueError(
                    f"Functions must be Callble, but {t} is of type {type(t)}"
                )

        if not isinstance(terminal_set, (list, dict)):
            raise ValueError(
                "Terminal set must be a list or a dictionary, "
                f"got {type(terminal_set)}."
            )

        # untyped case - convert to dictionary of Nones.
        if isinstance(terminal_set, list):
            # check if any function has type hints
            if any(f.__annotations__ for f in function_set):
                raise ValueError(
                    "Detected typed function with untyped terminal set. \
                    Please provide a dictionary with types for terminals."
                )

            terminal_set = {t: None for t in terminal_set}
            return function_set, terminal_set

        # typed case - check every value is a type
        for v in terminal_set.values():
            if not isinstance(v, type):
                raise ValueError(
                    "Values in terminal set dictionary must be types, "
                    f"but {v} is of type {type(v)}."
                )

        function_return_types = {
            get_return_type(f) for f in function_set
        }
        if root_type not in function_return_types:
            raise ValueError(
                f"Detected a mismatch between root_type ({root_type}) "
                f"and function set ({function_set}).\n"
                f"Root type must be the return type of at least one function."
            )
        if self.erc_type is not None and self.erc_type not in function_return_types:
            raise ValueError(
                f"Detected a mismatch between ERC type ({self.erc_type}) "
                f"and function set ({function_set}).\n"
                f"ERC range should not be defined if there are no numeric functions."
            )

        # check terminals and functions type intersection
        function_arg_types = {
            t for func in function_set for t in get_func_types(func)[:-1]
        }
        terminal_types = set(terminal_set.values())

        if self.erc_type:
            terminal_types.add(self.erc_type)

        if function_arg_types != terminal_types:
            raise ValueError(
                f"Function args type hints ({function_arg_types}) "
                f"must match terminal types ({terminal_types})."
            )

        return function_set, terminal_set

    def __str__(self) -> str:
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
        >>> tree.show()
        def func_0(x, y, z):
            return f_add(x, y)

        typed case:
        >>> tt = Tree(terminal_set={x: float, y: float},
        ...           function_set=[typed_add])
        >>> tt.tree = [FunctionNode(typed_add),
        ...            TerminalNode(1.0, float),
        ...            TerminalNode(2.0, float)]
        >>> str(tree)
        def func_0(x: float, y: float) -> float:
            return typed_add(1.0, 2.0)
        """
        args = (
            list(self.terminal_set.keys())
            if None in self.terminal_set.values()
            else [f"{k}: {v.__name__}" for k, v in self.terminal_set.items()]
        )
        ret_type_str = (
            ""
            if None in self.terminal_set.values()
            else f" -> {self.root.node_type.__name__}"
        )

        result = [
            f"def func_{str(self.id)}({', '.join(args)}){ret_type_str}:\n  return "
        ]
        self._str_rec("", [0], result)
        return "".join(result)

    def _str_rec(self, prefix: str, pos: List[int], result: str) -> str:
        """Recursively produce a simple textual printout of the tree
        (pos is a size-1 list so as to pass "by reference" on successive
         recursive calls).
        """

        node = self.tree[pos[0]]
        if isinstance(node, FunctionNode):
            result.append(f"{prefix}{str(node)}(\n")
            prefix = prefix + "  " if pos[0] == 0 else prefix
            for i in range(node.n_args):
                pos[0] += 1
                self._str_rec(prefix + "  ", pos, result)
                if i < node.n_args - 1:
                    result.append(",\n")
                elif 0 < i < node.n_args:
                    result.append("\n")
            result.append(prefix + ")")
        else:  # terminal
            result.append(f"{prefix}{str(node)}")

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

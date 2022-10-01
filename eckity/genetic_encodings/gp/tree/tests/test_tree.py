import pytest

from eckity.fitness.gp_fitness import GPFitness
from eckity.genetic_encodings.gp.tree.tree_individual import Tree, TerminalNode, FunctionNode
from eckity.genetic_encodings.gp.tree.functions import f_add, f_mul, f_sub


class TestTreeCase:
    """
    This class contains test cases for Tree and TreeNode classes
    """
    typed_functions = [(f_add, [int, int], int), (f_mul, [int, float], int), (f_sub, [int, int], int)]
    typed_terminals = [('x', int), ('y', int), (0, int), (1, int), (-1.0, float), (2.0, float)]
    untyped_functions = [f_add, f_mul, f_sub]
    untyped_terminals = ['x', 'y', 0, 1, -1]
    typed_tree = Tree(fitness=GPFitness(), root_type=int, function_set=typed_functions, terminal_set=typed_terminals)
    untyped_tree = Tree(fitness=GPFitness(), function_set=untyped_functions, terminal_set=untyped_terminals)

    @pytest.fixture
    def tear_down(self) -> None:
        """
        Empties both trees between tests
        Returns
        -------
        None
        """
        self.typed_tree.empty_tree()
        self.untyped_tree.empty_tree()

    def test_typed_untyped_exception(self, tear_down):
        """
        Tests an exception raises when the function and terminal sets don't match typed or untyped style
        Returns
        -------
        None
        """
        with pytest.raises(ValueError) as value:
            Tree(fitness=GPFitness(), function_set=self.typed_functions, terminal_set=self.untyped_terminals)
        assert "Tree received typed and untyped function and terminal sets!" == str(value.value)

        with pytest.raises(ValueError) as value:
            Tree(fitness=GPFitness(), function_set=self.untyped_functions, terminal_set=self.typed_terminals)
        assert "Tree received typed and untyped function and terminal sets!" == str(value.value)

    def test_root_type_mismatch(self, tear_down):
        """
        Tests a non-matching type node cannot be assigned as root not if it doesn't match the defined root_type
        Returns
        -------
        None
        """
        assert self.typed_tree.add_tree(TerminalNode(0.5, float)) is False
        assert self.typed_tree.add_tree(TerminalNode(0.5)) is False
        assert self.untyped_tree.add_tree(TerminalNode(0.5, float)) is False

    def tests_root_match(self, tear_down):
        """
        Tests a matching type node can be assigned as root not if it matches the defined root_type
        Returns
        -------
        None
        """
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.untyped_tree.add_tree(FunctionNode(f_add, 2, [None, None], None)) is True

    def tests_node_type_mismatch(self, tear_down):
        """
        Tests a non-matching type node cannot be assigned to the tree if it doesn't match the expected type
        Returns
        -------
        None
        """
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.typed_tree.add_tree(TerminalNode(True, bool)) is False
        assert self.typed_tree.add_tree(TerminalNode(True)) is False

        assert self.untyped_tree.add_tree(FunctionNode(f_add, 2, [None, None], None)) is True
        assert self.untyped_tree.add_tree(TerminalNode(True, bool)) is False

    def tests_node_type_match(self, tear_down):
        """
        Tests a matching type node can be assigned to the tree if it matches the expected type
        Returns
        -------
        None
        """
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.typed_tree.add_tree(TerminalNode(1, int)) is True
        assert 3 == self.typed_tree.size()

        assert self.untyped_tree.add_tree(FunctionNode(f_add, 2, [None, None], None)) is True
        assert self.untyped_tree.add_tree(TerminalNode(1)) is True
        assert 2 == self.untyped_tree.size()

    def test_execute(self, tear_down):
        """
        Tests right execution of the tree. In this case x+y+1==2 for x=1, y=0.
        Returns
        -------
        None
        """
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.typed_tree.add_tree(TerminalNode('x', int)) is True
        assert self.typed_tree.add_tree(TerminalNode('y', int)) is True
        assert self.typed_tree.add_tree(TerminalNode(1, int)) is True
        assert 2 == self.typed_tree.execute(x=1, y=0)

    def tests_replace_by_type(self, tear_down):
        """
        Test the correct replacement of a subtree with given one according to type constrains.
        Here we replace a subtree with float root consisting of only one terminal and validate the change using
        execution of the tree.
        -1*x+1==0 for x=1, y=0 changes to 2.0*x+1==3 for x=1, y=0.
        Returns
        -------
        None
        """
        assert self.typed_tree.add_tree(FunctionNode(f_add, 2, [int, int], int)) is True
        assert self.typed_tree.add_tree(FunctionNode(f_mul, 2, [int, float], int)) is True
        assert self.typed_tree.add_tree(TerminalNode('x', int)) is True
        assert self.typed_tree.add_tree(TerminalNode(-1.0, float)) is True
        assert self.typed_tree.add_tree(TerminalNode(1, int)) is True
        assert 0 == self.typed_tree.execute(x=1, y=0)

        assert self.typed_tree.replace_subtree([TerminalNode(2.0, float)]) is True
        assert 3 == self.typed_tree.execute(x=1, y=0)

        assert self.typed_tree.replace_subtree([TerminalNode(True, bool)]) is False
        assert self.typed_tree.replace_subtree([TerminalNode(1)]) is False

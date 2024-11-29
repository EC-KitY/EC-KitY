from typing import List

import pytest

from eckity.fitness import GPFitness
from eckity.genetic_encodings.gp import FunctionNode, TerminalNode, Tree, TreeNode
from eckity.genetic_operators import SubtreeCrossover


# Define custom functions
def typed_optional_inc(n: int, inc: bool) -> int:
    return n + 1 if inc else n


def untyped_optional_inc(n, inc):
    return n + 1 if inc else n


@pytest.mark.parametrize(
    "individuals, subtrees, expected",
    [
        (
            [
                Tree(
                    fitness=GPFitness(),
                    function_set=[typed_optional_inc],
                    terminal_set={True: bool, False: bool},
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(typed_optional_inc),
                        TerminalNode(1, int),
                        TerminalNode(True, bool),
                    ],
                    root_type=int,
                ),
                Tree(
                    fitness=GPFitness(),
                    function_set=[typed_optional_inc],
                    terminal_set={True: bool, False: bool},
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(typed_optional_inc),
                        TerminalNode(1, int),
                        TerminalNode(False, bool),
                    ],
                    root_type=int,
                ),
            ],
            [
                [
                    FunctionNode(typed_optional_inc),
                    TerminalNode(1, int),
                    TerminalNode(True, bool),
                ],
                [
                    FunctionNode(typed_optional_inc),
                    TerminalNode(1, int),
                    TerminalNode(False, bool),
                ],
            ],
            [
                [
                    FunctionNode(typed_optional_inc),
                    TerminalNode(1, int),
                    TerminalNode(False, bool),
                ],
                [
                    FunctionNode(typed_optional_inc),
                    TerminalNode(1, int),
                    TerminalNode(True, bool),
                ],
            ]
        ),
        (
            [
                Tree(
                    fitness=GPFitness(),
                    function_set=[typed_optional_inc],
                    terminal_set={True: bool, False: bool},
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(typed_optional_inc),
                        TerminalNode(1, int),
                        TerminalNode(True, bool),
                    ],
                    root_type=int,
                ),
                Tree(
                    fitness=GPFitness(),
                    function_set=[typed_optional_inc],
                    terminal_set={True: bool, False: bool},
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(typed_optional_inc),
                        TerminalNode(1, int),
                        TerminalNode(False, bool),
                    ],
                    root_type=int,
                ),
            ],
            [
                [
                    TerminalNode(True, bool),
                ],
                [
                    TerminalNode(False, bool),
                ],
            ],
            [
                [
                    FunctionNode(typed_optional_inc),
                    TerminalNode(1, int),
                    TerminalNode(False, bool),
                ],
                [
                    FunctionNode(typed_optional_inc),
                    TerminalNode(1, int),
                    TerminalNode(True, bool),
                ],
            ]
        ),
        ######################################################################
        # Untyped case
        ######################################################################
        (
            [
                Tree(
                    fitness=GPFitness(),
                    function_set=[untyped_optional_inc],
                    terminal_set=[True, False],
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(untyped_optional_inc),
                        TerminalNode(1),
                        TerminalNode(True),
                    ],
                ),
                Tree(
                    fitness=GPFitness(),
                    function_set=[untyped_optional_inc],
                    terminal_set=[True, False],
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(untyped_optional_inc),
                        TerminalNode(1),
                        TerminalNode(False),
                    ],
                ),
            ],
            [
                [
                    FunctionNode(untyped_optional_inc),
                    TerminalNode(1),
                    TerminalNode(True),
                ],
                [
                    FunctionNode(untyped_optional_inc),
                    TerminalNode(1),
                    TerminalNode(False),
                ],
            ],
            [
                [
                    FunctionNode(untyped_optional_inc),
                    TerminalNode(1),
                    TerminalNode(False),
                ],
                [
                    FunctionNode(untyped_optional_inc),
                    TerminalNode(1),
                    TerminalNode(True),
                ],
            ]
        ),
        (
            [
                Tree(
                    fitness=GPFitness(),
                    function_set=[untyped_optional_inc],
                    terminal_set=[True, False],
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(untyped_optional_inc),
                        TerminalNode(1),
                        TerminalNode(True),
                    ],
                ),
                Tree(
                    fitness=GPFitness(),
                    function_set=[untyped_optional_inc],
                    terminal_set=[True, False],
                    erc_range=(1, 3),
                    tree=[
                        FunctionNode(untyped_optional_inc),
                        TerminalNode(1),
                        TerminalNode(False),
                    ],
                ),
            ],
            [
                [
                    TerminalNode(True),
                ],
                [
                    TerminalNode(False),
                ],
            ],
            [
                [
                    FunctionNode(untyped_optional_inc),
                    TerminalNode(1),
                    TerminalNode(False),
                ],
                [
                    FunctionNode(untyped_optional_inc),
                    TerminalNode(1),
                    TerminalNode(True),
                ],
            ]
        ),
    ],
)
def test_swap_subtrees(
    individuals: List[Tree],
    subtrees: List[List[TreeNode]],
    expected: List[Tree],
):
    SubtreeCrossover._swap_subtrees(individuals, subtrees)
    assert [ind.tree for ind in individuals] == expected

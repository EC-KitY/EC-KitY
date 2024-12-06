import pytest

from eckity.base.untyped_functions import f_add
from eckity.creators import HalfCreator


def test_zero_group_size():
    half_creator = HalfCreator(
        init_depth=(2, 4), function_set=[f_add], terminal_set=["x"]
    )
    with pytest.raises(ValueError) as exc_info:
        half_creator.create_individuals(4, True)

    error_str = "Incompatible population size and init_depth."
    assert error_str in str(exc_info.value)


def test_create_individuals():
    half_creator = HalfCreator(
        init_depth=(3, 4), function_set=[f_add], terminal_set=["x"]
    )
    inds = half_creator.create_individuals(20, True)
    assert len(inds) == 20

    min_depth, max_depth = half_creator.init_depth
    assert all(min_depth <= ind.depth() <= max_depth for ind in inds)

    # assert at least 25% of the individuals are full trees of size 2^4
    # and another 25% are full trees of size 2^5 - 1
    sizes = [ind.size() for ind in inds]
    max_depth = half_creator.init_depth[1]
    assert sum(size == 2**max_depth - 1 for size in sizes) >= len(inds) / 4
    assert (
        sum(size == 2 ** (max_depth + 1) - 1 for size in sizes)
        >= len(inds) / 4
    )

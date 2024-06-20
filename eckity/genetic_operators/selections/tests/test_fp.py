import pytest

from eckity.fitness import SimpleFitness
from eckity.genetic_operators import FitnessProportionateSelection
from eckity.genetic_encodings.ga import BitStringVector


@pytest.mark.parametrize("higher_is_better", [False, True])
def test_selection_higher_is_better(higher_is_better):
    fp_sel = FitnessProportionateSelection(higher_is_better=higher_is_better)
    inds = [
        BitStringVector(
            SimpleFitness(1 / 1e6, higher_is_better=higher_is_better),
            length=4,
        ),
        BitStringVector(
            SimpleFitness(1e6, higher_is_better=higher_is_better), length=4
        ),
    ]
    result = fp_sel.select(inds, [])
    first_selected = result[0]

    # index of the expected individual to be selected
    expected_selected_idx = 1 if higher_is_better else 0

    assert first_selected.selected_by == [type(fp_sel).__name__]
    assert first_selected.cloned_from == [inds[expected_selected_idx].id]

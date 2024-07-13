import logging
from eckity.subpopulation import Subpopulation
from examples.treegp.basic_mode.symbolic_regression import (
    SymbolicRegressionEvaluator,
)


def test_zero_elites_warnning(caplog):
    with caplog.at_level(logging.WARNING):
        Subpopulation(
            SymbolicRegressionEvaluator(),
            population_size=10,
            elitism_rate=0.01,
        )
    assert len(caplog.records) == 1
    assert "elitism_rate" in caplog.text
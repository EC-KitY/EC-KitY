"""
Sanity tests for some basic examples.
"""

import pytest
import subprocess


def run_example(example_path: str):
    result = subprocess.run(
        ["python", "-m", example_path], capture_output=True
    )
    return result


@pytest.mark.parametrize(
    "module_path, n_reps, expected_fitness, higher_is_better",
    [
        ("examples.treegp.basic_mode.multiplexer", 3, 0.7, True),
    ],
)
def test_example(
    module_path: str,
    n_reps: int,
    expected_fitness: float,
    higher_is_better: bool,
):
    fitness_scores = []
    for _ in range(n_reps):
        # run the example
        result = subprocess.run(
            ["python", "-m", module_path],
            capture_output=True,
        )
        assert result.returncode == 0

        # parse best pure fitness:
        output = result.stdout.decode("utf-8")
        lines = output.split("\n")
        for line in lines:
            if "best pure fitness:" in line:
                best_pure_fitness = float(line.split(":")[-1].strip())
                fitness_scores.append(best_pure_fitness)

    mean_fitness = sum(fitness_scores) / n_reps

    if higher_is_better:
        assert mean_fitness >= expected_fitness
    else:
        assert mean_fitness <= expected_fitness

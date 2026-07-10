# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

EC-KitY is a Python evolutionary computation toolkit (Genetic Algorithm and tree-based Genetic Programming), designed to be scikit-learn compatible. It supports both a "basic" standalone mode (`evolve()` + `execute()`) and an sklearn-compatible mode (`fit()` + `predict()`) built on top of the same core.

## Commands

Install for development (from repo root):
```
pip install -r requirements.txt
pip install -r requirements-ml.txt --use-pep517   # optional: sklearn-compatible mode
```

Run the full test suite with coverage (mirrors CI):
```
coverage run -m pytest
coverage report --fail-under=85
```

Run a single test file or test:
```
pytest tests/test_sanity.py
pytest eckity/genetic_operators/crossovers/tests/test_subtree_crossover.py::test_name
```

Note: unit tests live next to the code they test (e.g. `eckity/genetic_operators/crossovers/tests/`, `eckity/genetic_encodings/gp/tree/tests/`), while `tests/` at the repo root holds cross-cutting/integration tests (sanity checks, reproducibility, multi-objective evolution).

Lint (as run in CI):
```
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
pylint $(git ls-files '*.py')   # see pylintrc for config; CI logs failures but does not fail the build
```

Run the examples (used in CI as smoke tests, requires `PYTHONPATH` set to repo root):
```
python examples/treegp/basic_mode/symbolic_regression.py
```

Generate API docs:
```
gendocs.bat
```

## Architecture

### Publisher/subscriber event system

Almost everything of consequence (`Algorithm`, `Operator`/genetic operators, `Breeder`) inherits from `BeforeAfterPublisher` (`eckity/before_after_publisher.py`) and/or `Operator` (`eckity/event_based_operator.py`). Each of these classes:
- Declares named events (e.g. `init`, `after_generation`, `before_operator`, `after_operator`, `before_eval`, `after_eval`) in its constructor via `event_names`.
- Implements `event_name_to_data(event_name)` to describe what payload is published for each event.
- Calls `self.publish(event_name)` at the right point, or wraps work in `act_and_publish_before_after(...)`.

External code (notably `Statistics` subclasses) hooks in via `register(event_name, callback)`. This is how statistics reporting, custom logging, and other cross-cutting behavior attach to the evolutionary run without modifying the core loop. When adding a new operator or algorithm phase, follow this pattern rather than adding ad-hoc callback params.

### Evolutionary run structure

`Algorithm` (`eckity/algorithms/algorithm.py`) is the abstract base for evolutionary runs; `SimpleEvolution` (`eckity/algorithms/simple_evolution.py`) is the single-subpopulation concrete implementation used by most examples. Key composition:

- **Population** (`eckity/population.py`) contains one or more **Subpopulation**s (`eckity/subpopulation.py`), each with its own `evaluator`, `creators` (with per-creator probabilities `pcr`), `operators_sequence`, `selection_methods`, and `elitism_rate`.
- **Breeder** (`eckity/breeders/`) drives one generation: applies elitism, then the first selection method, then runs `operators_sequence` over the selected individuals arity-by-arity. `SimpleBreeder` assumes a single subpopulation; multi-subpopulation algorithms need their own `Breeder` subclass.
- **PopulationEvaluator** (`eckity/evaluators/`) evaluates fitness across the population (concurrently, via `ProcessPoolExecutor`/`ThreadPoolExecutor` chosen by `executor=`) and returns the best individual.
- `Algorithm.evolve()` is the top-level entry point: `initialize()` (seed, executor, population creation, initial eval, publish `init`) → `evolve_main_loop()` (per-generation: set seed, `generation_iteration(gen)`, check `TerminationChecker`s, publish `after_generation`) → publish `evolution_finished` → `finish()`.
- Random seeding is per-generation and reproducible: `random_seed` seeds the run, `generation_seed` advances by 1 each generation (`next_seed()`), and both are pushed into a shared `RNG` (`eckity/random/`). This is what `tests/test_reproducibility.py` verifies.
- `Algorithm.__getstate__`/`__setstate__` special-case the `executor` field (a `ProcessPoolExecutor`/`ThreadPoolExecutor`) since it isn't picklable — recreate it from `_executor_type` rather than assuming state round-trips automatically.

### Individuals and genetic material

`Individual` (`eckity/individual.py`) is the abstract base for a candidate solution and owns a `Fitness` object (`eckity/fitness/`), lineage bookkeeping (`cloned_from`, `selected_by`, `applied_operators`, optional `parents`), and a monotonically increasing class-level `id` counter. Concrete encodings:
- GA vector encodings: `eckity/genetic_encodings/ga/` (`BitStringVector`, `FloatVector`, `IntVector`, base `VectorIndividual`).
- GP tree encoding: `eckity/genetic_encodings/gp/tree/` (`TreeIndividual`, `TreeNode`, `utils.py` for building terminal/function sets).

Genetic operators (`eckity/genetic_operators/`) are split into `crossovers/`, `mutations/`, and `selections/`, and each concrete operator implements `apply_operator(individuals)` from the base `Operator`/`GeneticOperator` classes. Many operators (crossover, mutation) also use `FailableOperator` (`failable_operator.py`) to retry/report failure when an operator can't be legally applied (e.g. tree too deep).

### Sklearn-compatible mode

`eckity/sklearn_compatible/` wraps an `Algorithm` in a `SklearnWrapper` (`BaseEstimator`) exposing `fit`/`predict`; `SKRegressor` and `SKClassifier` are the concrete estimators, paired with `RegressionEvaluator`/`ClassificationEvaluator` (`IndividualEvaluator` subclasses that read `(X, y)` context set via `evaluator.set_context(...)` in `fit`). This is an additive layer — the same `Algorithm`/`Population`/`Individual` objects work in both modes; only the fitting/prediction entry point differs (`algo.evolve()` + `algo.execute(**kwargs)` vs `regressor.fit(X, y)` + `regressor.predict(X)`).

### Multi-objective

`eckity/multi_objective_evolution/` adds non-dominated-front selection and related machinery on top of the same `Subpopulation`/`Breeder` abstractions above, rather than a parallel architecture — see `tests/moe_test/` and `examples/multi_objective/` for usage patterns.

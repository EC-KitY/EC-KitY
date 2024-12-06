# Evaluator

The Evaluator is responsible for computing the fitness values of the individuals in the experiment.

There are two types of evaluators:
1) PopulationEvaluator
2) IndividualEvaluator

## Population Evaluator

Responsible of computing fitness in population-level.

Currently, the only concrete PopulationEvaluator subclass is `SimplePopulationEvaluator`, that assumes fitness scores of distinct individuals are independent.

Since fitness computation is costly, it is performed concurrently using **Executors**.

Currently, we support `ThreadPoolExecutor` and `ProcessPoolExecutor`.
The documentation of Executors is available [here](https://docs.python.org/3/library/concurrent.futures.html).

## Process Pool Executor
This executor computes fitness scores by creating multiple processes that can run on different CPUs in parallel. Note that this adds a slight overhead in runtime since process creation and inter-process communication is costly. Thus, this executor should be used for heavier fitness functions.


## Thread Pool Executor
This executor creates several threads for fitness computation. Due to Python's GIL (Global Interpreter Lock), only a single thread can be executed at a time.
This executor should be used if your fitness function releases the GIL frequently.
GIL is released by performing I/O operations (opening a file, reading from a socket, etc.), or by invoking C functions such as in `NumPy`([NumPy vs GIL reference](https://superfastpython.com/numpy-vs-gil/)).


## Max Workers
The value of `max_workers` dictates the number of worker processes/threads created. Passing `None` will automatically assign the maximum possible number (which depends on your hardware). Note that setting a high number of workers will lead to a large amount of context switches between them, which can cause a large overhead.

## Usage
Setting the desired Executor is done by passing the (optional) arguments `executor` and `max_workers` to the constructor of `SimpleEvolution`, as follows:
```python
algo = SimpleEvolution(
    # ...
    executor="process",  # or "thread"
    max_workers=3,
    # ...
)
```

## Individual Evaluator
Responsible of computing the fitness value of a single individual.

Currently, the only concrete IndividualEvaluator subclass is `SimpleIndividualEvaluator`, that computes the fitness score of each individual independently.

`SimpleIndividualEvaluator` defines an abstract `evaluate_individual` method. This method receives a single individual and returns a float value representing its fitness score.

Each Evolutionary Computation problem should have a dedicated SimpleIndividualEvaluator subclass that will compute the specific fitness function of the problem.

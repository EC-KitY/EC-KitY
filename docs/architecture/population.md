# Population

The Individual class represents the population. It consists of one or more Subpopulations.

# Subpopulation

This class represents a collection of individuals.
We usually refer to the simple case, where only one subpopulation exists in the entire evolutionary algorithm.

Each Subpopulation may have its own fitness evaluation function (defined by the [IndividualEvaluator](architecture/evaluator.md)).
There are additional parameters of Subpopulation (e.g., [genetic operators](architecture/operators.md)), and the algorithm (e.g., [Breeder](architecture/breeder.md) and [Statistics](architecture/statistics.md)).

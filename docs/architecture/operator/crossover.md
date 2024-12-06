# Crossover

Crossover is performed during the breeding process and combines the genotype of several individuals **in-place**.

The exact amount of operated individuals is determined by the `arity` field.
Due to the implementation of SimpleBreeder, the amount of returned individuals **must** be equal to the amount of operated individuals.
For instance, a subtree crossover performed between two GP trees must return two offsprings.

If you must include an operator with a different number of parents and offsprings, you can override the implementation of the breeder.

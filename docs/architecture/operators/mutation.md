# Mutation

Mutation is performed during the breeding process and independently alters the genotype of one or more individuals **in-place**.

The exact amount of operated individuals is determined by the `arity` field.
Due to the implementation of SimpleBreeder, the amount of returned individuals **must** be equal to the amount of operated individuals.
For instance, a subtree mutation performed independently on 10 trees at once must return 10 GP trees.

If you must include an operator with a different number of parents and offsprings, you can override the implementation of the breeder.

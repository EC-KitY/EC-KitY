# Breeder

The Breeder is responsible for the parent selection and the execution of genetic operators (crossover, mutation) in each generation.

## SimpleBreeder

SimpleBreeder assumes there is only one Subpopulation, and that distinct individuals are independent of each other.

## The flow of SimpleBreeder

SimpleBreeder first selects all the parents from the population of the previous generation, then applies the operators sequence in a pipeline manner.

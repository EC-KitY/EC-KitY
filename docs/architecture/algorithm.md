# Algorithm

Algorithm manages the Evolutionary Algorithm.

## Algorithm types

Algorithm is an abstract class.
Currently, the only concrete Algorithm class is *SimpleEvolution*.
SimpleEvolution assumes there is only one population in the evolutionary experiment (i.e. no coevolution).

An algorithm can operate on a single Subpopulation (as in the case of
SimpleEvolution), two Subpopulations (e.g., in a coevolutionary setup), or
n (> 2) Subpoplations (e.g., island model).

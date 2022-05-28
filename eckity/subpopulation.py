import numpy as np

from eckity.creators.creator import Creator
from eckity.creators.gp_creators.full import FullCreator
from eckity.creators.gp_creators.tree_creator import GPTreeCreator

from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.erc_mutation import ERCMutation
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection

import random


class Subpopulation:
    """
    Sub-group of the experiment population.

    Contains a specific encoding, fitness evaluation method, creator list, operator sequence and selection methods.

    Parameters
    ----------
    evaluator: IndividualEvaluator
        fitness evaluation method for the individuals of this sub-population

    creators: Creator or list of Creators, default=None
        possible creators to generate individuals according to the encoding
        of this sub-population (GPTrees, Bit Vectors etc.)

    pcr: list of integers, default=None
        probability mapping for each creator in creators parameter.
        Length must match the length of creators parameter.

    operators_sequence: list of Crossovers and Mutations, default=None
        Possible crossover and mutation actions that can change the individuals' representations.
        The operators will be done sequentially in each generation, by their order in the list.
        See eckity.genetic_operators for more details on crossover and mutation operators

    selection_methods: list of SelectionMethods
        Methods for selecting individuals in each generation.
        See eckity.genetic_operators for more details on selection methods

    elitism_rate: float, default=0.0
        What percentage of the sub-population's individuals should be kept as elites for the next generation

    population_size: int, default=200
        The number of individuals in this sub-population.

    individuals: list of Individuals, default=None
        The individuals list of this sub-population.

    higher_is_better: bool, default=False
        Determines if the fitness value of this sub-population's individuals should be maximized or minimized.

    Attributes
    ----------
    n_elite: int
        Number of the sub-population's elite individuals.
        In every generation, there will be n_elites slots for the elite individuals
        that will be copied as-is to the next generation.
    """
    def __init__(self,
                 evaluator,
                 creators=None,
                 pcr=None,
                 operators_sequence=None,
                 selection_methods=None,
                 elitism_rate=0.0,
                 population_size=200,
                 individuals=None,
                 higher_is_better=False):

        # verify valid creators and creation probability inputs
        if creators is None:
            full_cr = FullCreator()
            creators = [full_cr]
        elif isinstance(creators, Creator):
            creators = [creators]
        elif isinstance(creators, list):
            if len(creators) == 0:
                raise ValueError('Creators list cannot be empty')
            for creator in creators:
                if not isinstance(creator, Creator):
                    raise ValueError('Detected a non-creator instance as an element in creators list')
        else:
            raise ValueError(
                'Parameter creators must be either a Creator or a list of Creators\n '
                'received creators with unexpected type of', type(creators)
            )

        if pcr is None:
            pcr = [1 / len(creators) for _ in creators]

        if len(creators) != len(pcr):
            raise ValueError(f'Number of creators ({len(creators)}) \
                                    must match number of creation probabilities {(len(pcr))}!')
        if sum(pcr) != 1:
            raise ValueError(f'Sum of creation probabilities ({pcr}) must be 1!')

        # set default args
        if operators_sequence is None:
            operators_sequence = [SubtreeCrossover(arity=2, probability=0.9),
                                  SubtreeMutation(arity=1, probability=0.7),
                                  ERCMutation(arity=1, probability=0.1)]
        if selection_methods is None:
            selection_methods = [(TournamentSelection(tournament_size=10, higher_is_better=higher_is_better), 1)]

        self.creators = creators
        self._pcr = pcr
        self._operators_sequence = operators_sequence
        self.population_size = population_size
        self._selection_methods = selection_methods
        self.higher_is_better = higher_is_better
        self.evaluator = evaluator

        self.n_elite = round(elitism_rate * self.population_size)
        self.individuals = individuals

    def create_subpopulation_individuals(self):
        if self.individuals is None:
            # Select one creator to generate individuals, with respect to the creators' probabilities
            selected_creator = random.choices(self.creators, self._pcr)[0]  # random.choices returns [selected_creator]
            self.individuals = selected_creator.create_individuals(self.population_size, self.higher_is_better)

    def get_operators_sequence(self):
        return self._operators_sequence

    def get_selection_methods(self):
        return self._selection_methods

    def get_best_individual(self):
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.get_augmented_fitness(),
                             reverse=self.higher_is_better)
        return sorted_inds[0]

    def get_worst_individual(self):
        sorted_inds = sorted(self.individuals, key=lambda ind: ind.get_augmented_fitness(),
                             reverse=not self.higher_is_better)
        return sorted_inds[0]

    def get_average_fitness(self):
        return np.mean([indiv.get_pure_fitness() for indiv in self.individuals])

    def contains_individual(self, individual):
        return individual in self.individuals

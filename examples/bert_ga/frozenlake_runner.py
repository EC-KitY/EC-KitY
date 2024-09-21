import sys

import numpy as np
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.ga_creators.bit_string_vector_creator import (
    GABitStringVectorCreator,
)
from eckity.creators.ga_creators.int_vector_creator import GAIntVectorCreator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import (
    VectorKPointsCrossover,
)
from eckity.genetic_operators.mutations.vector_random_mutation import (
    BitStringVectorNFlipMutation,
    IntVectorNPointMutation,
)
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.subpopulation import Subpopulation
from plot_statistics import PlotStatistics

from problems.blackjack.blackjack_evaluator import BlackjackEvaluator
from problems.frozen_lake.frozen_lake_evaluator import FrozenLakeEvaluator, FROZEN_LAKE_STATES
from problems.monster_cliff_walking.monstercliff_evaluator import (
    MonsterCliffWalkingEvaluator,
)

length = FROZEN_LAKE_STATES
creator = GAIntVectorCreator(length=length, bounds=(0, 3))
ind_eval = FrozenLakeEvaluator(total_episodes=2000)
mutation = IntVectorNPointMutation(probability=0.3, n=length // 10)
generations = 50

evo = SimpleEvolution(
        Subpopulation(
            creators=creator,
            population_size=100,
            evaluator=ind_eval,
            higher_is_better=True,
            elitism_rate=0.0,
            operators_sequence=[
                VectorKPointsCrossover(probability=0.7, k=2),
                mutation,
            ],
            selection_methods=[
                # (selection method, selection probability) tuple
                (
                    TournamentSelection(
                        tournament_size=4, higher_is_better=True
                    ),
                    1,
                )
            ],
        ),
        breeder=SimpleBreeder(),
        max_generation=generations,
        # executor="process",
        # max_workers=None,
        statistics=PlotStatistics(),

    )
evo.evolve()
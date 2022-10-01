import csv
import random
from time import time

import numpy as np

from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.creators.gp_creators.full import FullCreator
from eckity.genetic_encodings.gp.tree.functions import f_and, f_or, f_not, f_if_then_else, f_add, f_mul, f_sub, f_div, \
    f_equal, f_lt
from eckity.genetic_operators.crossovers.subtree_crossover import SubtreeCrossover
from eckity.genetic_operators.mutations.subtree_mutation import SubtreeMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from examples.treegp.non_sklearn_mode.spambase_problem.spambase_evaluator import SpambaseEvaluator


def main():
    """
    This problem is a classification example using STGP (Strongly Typed Genetic Programming). The evolved programs work
    on floating-point values AND Booleans values. The programs must return a Boolean value which must be true if e-mail
    is spam, and false otherwise. It uses a base of emails (saved in spambase.csv, see Reference), from which it
    randomly chooses 400 emails to evaluate each individual.
    References
    ----------
    DEAP Spambase Problem Example: https://deap.readthedocs.io/en/master/examples/gp_spambase.html
    """
    start_time = time()

    terminal_set = [(True, bool), (False, bool)] + [(i, float) for i in np.random.uniform(0, 100, 100)] + \
                   [(f'v{i}', float) for i in range(57)]

    function_set = [(f_and, [bool, bool], bool), (f_or, [bool, bool], bool), (f_not, [bool], bool),
                    (f_if_then_else, [bool, float, float], float), (f_add, [float, float], float),
                    (f_mul, [float, float], float), (f_sub, [float, float], float), (f_div, [float, float], float),
                    (f_equal, [float, float], bool), (f_lt, [float, float], bool)]

    # Initialize SimpleEvolution instance
    algo = SimpleEvolution(
        Subpopulation(creators=FullCreator(root_type=bool,
                                           init_depth=(2, 4),
                                           terminal_set=terminal_set,
                                           function_set=function_set,
                                           bloat_weight=0.00001),
                      population_size=100,
                      # user-defined fitness evaluation method
                      evaluator=SpambaseEvaluator(),
                      # this is a maximization problem (fitness is accuracy), so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=0.05,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          SubtreeCrossover(probability=0.3, arity=2),
                          SubtreeMutation(probability=0.9, arity=1)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=3, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=1,
        max_generation=50,
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=400, threshold=20),
        statistics=BestAverageWorstStatistics(),
        random_seed=10
    )
    # evolve the generated initial population
    algo.evolve()

    # execute the best individual after the evolution process ends

    results = []
    with open("spambase.csv") as spambase:
        spamReader = csv.reader(spambase)
        spam = list(list(float(elem) for elem in row) for row in spamReader)
    mails = random.sample(spam, 50)
    for mail in mails:
        results.append(
            bool(algo.execute(v0=mail[0], v1=mail[1], v2=mail[2], v3=mail[3], v4=mail[4], v5=mail[5], v6=mail[6],
                              v7=mail[7], v8=mail[8], v9=mail[9], v10=mail[10], v11=mail[11], v12=mail[12],
                              v13=mail[13], v14=mail[14], v15=mail[15], v16=mail[16], v17=mail[17], v18=mail[18],
                              v19=mail[19], v20=mail[20], v21=mail[21], v22=mail[22], v23=mail[23], v24=mail[24],
                              v25=mail[25], v26=mail[26], v27=mail[27], v28=mail[28], v29=mail[29], v30=mail[30],
                              v31=mail[31], v32=mail[32], v33=mail[33], v34=mail[34], v35=mail[35], v36=mail[36],
                              v37=mail[37], v38=mail[38], v39=mail[39], v40=mail[40], v41=mail[41], v42=mail[42],
                              v43=mail[43], v44=mail[44], v45=mail[45], v46=mail[46], v47=mail[47], v48=mail[48],
                              v49=mail[49], v50=mail[50], v51=mail[51], v52=mail[52], v53=mail[53], v54=mail[54],
                              v55=mail[55], v56=mail[56])) is bool(mail[57]))

    print("Correct predictions: {}".format(str(sum(results)/len(results))))

    print('total time:', time() - start_time)


if __name__ == '__main__':
    main()

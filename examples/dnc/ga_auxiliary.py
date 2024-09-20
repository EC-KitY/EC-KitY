import json
import os
import random
from copy import deepcopy

import numpy as np


def read_float_list_from_text_file(path: str) -> list:
    """
    :param path: path to a text file
    :return: list of strings
    """
    with open(path, 'r') as f:
        return [float(l) for l in f.readlines()]


def create_output_folder(path: str) -> None:
    """
    Creates an output folder in the given path
    :param path:
    :return:
    """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def read_dict(name: str):
    """
    :param name: path to a json dict
    """
    with open(name, 'r') as f:
        return json.load(f)


def save_dict(dct, path: str) -> None:
    """
    Saves a dictionary in json-form, so results can be read mid-experiment.
    :param dct: Some dictionary
    :param path: path to save
    :return: None
    """
    with open(path, "w") as f:
        json.dump(dct, f)


def save_list(lst, path: str) -> None:
    """
    Saves a list in text-form, so results can be read mid-experiment.
    :param lst: Some list
    :param path: path to save
    :return: None
    """
    with open(path, "w") as f:
        for s in lst:
            f.write(str(s) + "\n")


def generate_child_n_point(ind1, ind2, parent_selection_prob=0.5):
    """
    n point Crossover between two individuals
    :param ind1: GA Parent 1
    :param ind2: GA Parent 2
    :param parent_selection_prob: Probability to select a gene from parent 1
    :return: Children
    """
    child = []

    for i in range(len(ind1)):
        if random.random() < parent_selection_prob:
            child += [ind1[i]]
        else:
            child += [ind2[i]]

    return child


def n_point_crossover(ind1, ind2):
    return generate_child_n_point(ind1, ind2), generate_child_n_point(ind1, ind2)


def is_in_dict(ind, fitness_dict):
    """
    :param fitness_dict:
    :param ind: GA individual
    :return: True iff the given individual has a value store in the cache (past metric)
    """
    code = tuple(ind)
    return code in fitness_dict


def get_fit(ind, fitness_dict):
    """
    :param fitness_dict:
    :param ind: GA individual
    :return: Saved metric
    """
    code = tuple(ind)
    if code in fitness_dict:
        return fitness_dict[code]


def save_fitness(ind, val, fitness_dict):
    """
    :param fitness_dict:
    :param ind: GA individual to save
    :param val: metric to save
    :return: None
    """
    code = tuple(ind)
    fitness_dict[code] = val


def n_point_crossover_pairs(pairs_to_cross, crossover_func=None):
    """
    :param crossover_func: crossover
    :param pairs_to_cross: list of pairs to cross
    :return: list of crossed pairs
    """
    crossed_pairs = []
    for pair in pairs_to_cross:

        if crossover_func is not None:
            crossed_pairs += [crossover_func(pair[0], pair[1])]
        else:
            crossed_pairs += [n_point_crossover(pair[0], pair[1])]

    return crossed_pairs


def n_point_mutate(individual, mutate_prob, min_val, max_val):
    """
    :param individual: Individual to mutate
    :param mutate_prob: Probability to mutate each gene
    :param min_val: Minimum value for a gene
    :param max_val: Maximum value for a gene
    :return: Mutated individual
    """
    mutation_mask = np.random.choice([0, 1], size=len(individual), p=[1 - mutate_prob, mutate_prob])
    individual[mutation_mask == 1] = np.random.randint(min_val, max_val + 1, size=np.sum(mutation_mask == 1))
    return individual


def choose_from_competition(competition, fitness_dict):
    """
    :param competition: list of individuals
    :param fitness_dict: fitness dict
    :return: winner of the competition by fitness
    """
    fitnesses = [fitness_dict[tuple(ind)] for ind in competition]
    return np.copy(competition[np.argmax(fitnesses)])


def tournament_selection(individuals, how_many_to_select, tournament_size, fitness_dict):
    """
    :param fitness_dict: fitness cache
    :param individuals: Population
    :param how_many_to_select: How many individuals to select
    :param tournament_size: How many individuals per tournament
    :return: k selected individuals
    """
    tournament_indexes = np.random.randint(0, len(individuals), size=(how_many_to_select, tournament_size))
    tournaments = [individuals[tournament_index] for tournament_index in tournament_indexes]
    selected_individuals = [choose_from_competition(tournament, fitness_dict) for tournament in
                            tournaments]

    return deepcopy(np.array(selected_individuals))

import copy
import os
import random
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from ga_auxiliary import create_output_folder, save_list, save_dict, is_in_dict, get_fit, \
    save_fitness, n_point_crossover_pairs, n_point_mutate, tournament_selection


class SelectionGA:

    def __init__(self, n_generations: int, population_size: int, crossover_prob: float, mutation_prob: float,
                 ind_length: int, min_selection_val, max_selection_val, random_state: int = 42, tournament_size=5,
                 save_every_n_generations: int = 10, crossover_func: callable = n_point_crossover_pairs,
                 flip_mutation_prob=0.005, save_population_info=False,
                 save_fitness_info=False, elitism=False, n_parents=2):
        """
        :param n_generations: Number of generations to run
        :param population_size: Population Size
        :param crossover_prob: Crossover probability
        :param mutation_prob:  Mutation probability
        :param ind_length: Individual length
        :param random_state: Initial random seed
        """
        assert 0 <= crossover_prob <= 1, "ILLEGAL CROSSOVER PROBABILITY"
        assert 0 <= mutation_prob <= 1, "ILLEGAL MUTATION PROBABILITY"
        assert population_size > 1, "Population size must be at least 2"
        assert n_generations > 0, "Number of generations must be a positive integer"
        assert ind_length > 0, "Illegal individual length"

        # params
        self.n_generations = n_generations
        self.population_size = population_size
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.random_state = random_state
        self.save_every_n_generations = save_every_n_generations
        self.flip_mutation_prob = flip_mutation_prob
        self.min_selection_val = min_selection_val
        self.max_selection_val = max_selection_val
        self.tournament_size = tournament_size
        self.crossover_func = crossover_func
        self.ind_length = ind_length
        self.save_population_info = save_population_info
        self.save_fitness_info = save_fitness_info
        self.elitism = elitism
        self.n_parents = n_parents
        # params

        self.fitness_dict = {}
        self.pop_dict = {}

        self.generation_metrics = {
            'std': [],
            'mean': [],
            'median': [],
            'max': [],
            'min': [],
            'time': []
        }

    def reset_stats(self) -> None:
        """
        Resets saved-metrics.
        :return: None
        """
        self.fitness_dict = {}
        self.pop_dict = {}
        for key in self.generation_metrics.keys():
            self.generation_metrics[key] = []

    def update_progress_arrays(self, fits, g, gen_start, output_path):
        """
        :param fits: list of fitness scores
        :param g: current generation number
        :param gen_start: time that this generation started
        :param output_path: output to save the data
        :return: None
        """
        generation_time = time.time() - gen_start
        fits = np.array(fits)

        self.generation_metrics['mean'] += [fits.mean()]
        self.generation_metrics['std'] += [fits.std()]
        self.generation_metrics['median'] += [np.median(fits)]
        self.generation_metrics['max'] += [np.max(fits)]
        self.generation_metrics['min'] += [np.min(fits)]
        self.generation_metrics['time'] += [generation_time]

        self.save_all_data(g, output_path)

    def save_all_data(self, curr_generation: int, output_folder: str) -> None:
        """
        Saves metrics for the given generations in a given path.
        :param curr_generation: Current generation (int)
        :param output_folder: folder to save data in.
        :return: None
        """
        with open(output_folder + "am_alive", "w") as f:
            f.write("Still running at generation :" + str(curr_generation) + "\n")

        is_last_gen = (curr_generation == (self.n_generations - 1))
        if (curr_generation % self.save_every_n_generations != 0) and not is_last_gen:
            return

        print(f"saving data for {curr_generation=}")

        if is_last_gen:
            if self.save_fitness_info:
                written_dict = {str(k): v.tolist() for k, v in self.fitness_dict.items()}
                save_dict(written_dict, output_folder + "fitness_dict.json")

            if self.save_population_info:
                save_dict({gen_num: pop.tolist() for gen_num, pop in self.pop_dict.items()},
                          output_folder + "gens_dict.json")

        for metric_name, metric_list in self.generation_metrics.items():
            save_list(metric_list, output_folder + metric_name + ".txt")

    def init_population(self, length_to_gen: int, population_size):
        """
        Creates a random individual with a given length
        :param population_size:
        :param length_to_gen: How many '1/0' to generate
        :return: A random binary array of the given size
        """
        return np.random.randint(self.min_selection_val, self.max_selection_val + 1,
                                 size=(population_size, length_to_gen))

    def __init_seeds(self) -> None:
        """
        Resets seeds back to the initial seed
        :return: None
        """
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        os.environ['PYTHONHASHSEED'] = str(self.random_state)

    def fit(self, output_folder, fitness_func, crossover_func=None, stopping_func=None):
        self.reset_stats()
        self.__init_seeds()

        if crossover_func is not None:
            self.crossover_func = crossover_func

        create_output_folder(output_folder)

        self.__run_gens(output_folder, fitness_func, stopping_func=stopping_func)

    def crossover(self, population, crossover_prob):
        """
        Apply crossover and mutation on the offsprings
        """
        pairs_to_cross = []
        next_gen = []

        crossover_masks = np.random.uniform(size=len(population) // 2)
        extended_crossover_parent_indexes = np.random.randint(low=0, high=len(population),
                                                              size=(len(population) // 2, self.n_parents - 2))
        crossover_index = 0
        for child1, child2, cross_mask in zip(population[::2], population[1::2], crossover_masks):
            if cross_mask < crossover_prob:
                current_tuple = [child1.copy(), child2.copy()]
                if self.n_parents > 2:
                    current_tuple += [population[extended_crossover_parent_indexes[crossover_index, i]].copy() for i in
                                      range(extended_crossover_parent_indexes.shape[1])]
                pairs_to_cross.append(current_tuple)
            crossover_index += 1

        crossed_parents_pairs = self.crossover_func(pairs_to_cross)

        for child1, child2, cross_mask in zip(population[::2], population[1::2], crossover_masks):
            if cross_mask < crossover_prob:
                next_gen += crossed_parents_pairs.pop(0)
            else:
                next_gen += [child1.copy(), child2.copy()]

        return np.array(next_gen)

    def mutate(self, population, mutation_prob):
        """
        Apply mutation on the offsprings
        """
        next_gen = []
        for ind in population:
            if np.random.uniform() < mutation_prob:
                next_gen_ind = self.mutate_individual(ind)
            else:
                next_gen_ind = ind.copy()

            next_gen.append(next_gen_ind)

        return np.array(next_gen)

    def mutate_individual(self, ind):
        """
        Mutate a single individual
        """
        return n_point_mutate(ind.copy(), self.flip_mutation_prob, self.min_selection_val,
                              self.max_selection_val)

    def evaluate_and_set_fitness(self, population, fitness_func):
        """
        Evaluate and set fitness for the given population
        """
        fitness_values = [fitness_func(ind, self.fitness_dict) for ind in tqdm(population)]
        for ind, fit in zip(population, fitness_values):
            save_fitness(ind, fit, self.fitness_dict)

    def select(self, population):
        """
        Select the next generation
        """
        num_elitists = int(self.elitism)
        next_gen = tournament_selection(population, len(population) - num_elitists, self.tournament_size,
                                        self.fitness_dict)

        if num_elitists > 0:
            elitists = copy.deepcopy(np.array([ind for ind, _ in
                                               sorted(self.fitness_dict.items(), key=lambda x: x[1], reverse=True)[
                                               :num_elitists]]))
            next_gen = np.concatenate((next_gen, elitists), axis=0)

        return next_gen

    def __run_gens(self, output_folder: str, fitness_func: callable, stopping_func: callable = None):
        """
        :param output_folder: Folder to save output data
        :param fitness_func: Function that receives an individual and returns its fitness
        :param stopping_func: Function that receives the current fitness and returns whether to stop
        :return: None
        """
        population = self.init_population(length_to_gen=self.ind_length, population_size=self.population_size)
        self.evaluate_and_set_fitness(population, fitness_func)
        start_time = time.time()
        for generation_index in tqdm(range(0, self.n_generations)):
            print("-- Generation %i --" % generation_index)

            if self.save_population_info:
                self.pop_dict[generation_index] = deepcopy(population)

            offspring = self.select(population)
            fitness_values = [get_fit(ind, self.fitness_dict) for ind in offspring]

            if stopping_func is not None and stopping_func(fitness_values):
                print('Stopping after %i generations' % generation_index)
                break

            print('generation %i, best fitness: %f' % (generation_index, np.max(fitness_values)))
            self.update_progress_arrays(fitness_values, generation_index, start_time, output_folder)
            start_time = time.time()
            offspring = self.crossover(offspring, self.crossover_prob)
            offspring = self.mutate(offspring, self.mutation_prob)

            invalid_individuals = [ind for ind in offspring if not is_in_dict(ind.tolist(), self.fitness_dict)]
            self.evaluate_and_set_fitness(invalid_individuals, fitness_func)
            population = offspring

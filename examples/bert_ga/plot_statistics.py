from sys import stdout

import matplotlib.pyplot as plt
import numpy as np
from eckity.statistics.statistics import Statistics


class PlotStatistics(Statistics):
    """
    Concrete Statistics class.
    Provides statistics about the best fitness, average fitness and worst fitness of every sub-population in
    some generation.

    Parameters
    ----------
    format_string: str
        String format of the data to output.
        Value depends on the information the statistics provides.
        For more information, check out the concrete classes who extend this class.

    output_stream: Optional[SupportsWrite[str]], default=stdout
        Output file for the statistics.
        By default, the statistics will be written to stdout.
    """

    def __init__(self, format_string=None, output_stream=stdout):
        super().__init__(format_string)
        self.mean_fitnesses = []
        self.median_fitnesses = []
        self.best_fitnesses = []
        self.model_errors = []

    def write_statistics(self, sender, data_dict):
        sub_pop = data_dict["population"].sub_populations[0]

        fitnesses = [ind.get_pure_fitness() for ind in sub_pop.individuals]
        fitnesses = np.array(fitnesses)
        self.mean_fitnesses.append(np.mean(fitnesses))
        self.median_fitnesses.append(np.median(fitnesses))

        hib = sub_pop.higher_is_better
        best_fitness = np.max(fitnesses) if hib else np.min(fitnesses)
        self.best_fitnesses.append(best_fitness)

        pop_eval = sender.population_evaluator
        if hasattr(pop_eval, "approx_fitness_error"):
            self.model_errors.append(pop_eval.approx_fitness_error)

    def plot_statistics(self, *args):
        assert (
            len(self.mean_fitnesses)
            == len(self.median_fitnesses)
            == len(self.best_fitnesses)
        ), "Statistics lists are not the same length"

        print("mean_fitnesses =", self.mean_fitnesses)
        print("median_fitnesses =", self.median_fitnesses)
        print("best_fitnesses =", self.best_fitnesses)

        plt.title(str(*args))
        plt.plot(self.mean_fitnesses, label="mean")
        plt.plot(self.median_fitnesses, label="median")
        plt.plot(self.best_fitnesses, label="best")
        plt.xlabel("generation")
        plt.ylabel("fitness")
        plt.xticks(range(0, len(self.mean_fitnesses) + 1, 5))
        plt.legend()
        plt.show()

    # Necessary for valid pickling, since modules cannot be pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["output_stream"]
        return state

    # Necessary for valid unpickling, since modules cannot be pickled
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.output_stream = stdout
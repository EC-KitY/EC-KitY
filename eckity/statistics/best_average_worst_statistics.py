from sys import stdout

from eckity.statistics.statistics import Statistics


class BestAverageWorstStatistics(Statistics):
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
        if format_string is None:
            format_string = 'best fitness {}\nworst fitness {}\naverage fitness {}\n'
        super().__init__(format_string, output_stream)

    def write_statistics(self, sender, data_dict):
        print(f'generation #{data_dict["generation_num"]}', file=self.output_stream)
        for index, sub_pop in enumerate(data_dict["population"].sub_populations):
            print(f'subpopulation #{index}', file=self.output_stream)
            best_individual = sub_pop.get_best_individual()
            print(self.format_string.format(best_individual.get_pure_fitness(),
                                            sub_pop.get_worst_individual().get_pure_fitness(),
                                            sub_pop.get_average_fitness()), file=self.output_stream)
            print(best_individual.vector)

    # TODO tostring to indiv

    # Necessary for valid pickling, since modules cannot be pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['output_stream']
        return state

    # Necessary for valid unpickling, since modules cannot be pickled
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.output_stream = stdout

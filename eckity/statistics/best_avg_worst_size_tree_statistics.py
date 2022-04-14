import sys

import numpy as np

from eckity.statistics.statistics import Statistics


class BestAverageWorstSizeTreeStatistics(Statistics):
    def __init__(self, format_string=None, output_stream=sys.stdout):
        if format_string is None:
            format_string = 'best fitness {}\nworst fitness {}\naverage fitness {}\naverage size {}\n'
        super().__init__(format_string, output_stream)

    def write_statistics(self, sender, data_dict):
        print(f'generation #{data_dict["generation_num"]}', file=self.output_stream)
        for index, sub_pop in enumerate(data_dict['population'].sub_populations):
            print(f'subpopulation #{index}', file=self.output_stream)
            best_individual = sub_pop.get_best_individual()
            print(
                self.format_string.format(best_individual.get_pure_fitness(),
                                          sub_pop.get_worst_individual().get_pure_fitness(),
                                          sub_pop.get_average_fitness(),
                                          np.average([ind.size() for ind in sub_pop.individuals])),
                file=self.output_stream
            )

    # TODO tostring to indiv

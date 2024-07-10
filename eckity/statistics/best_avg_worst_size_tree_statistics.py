import logging

import numpy as np

from eckity.statistics.statistics import Statistics

logger = logging.getLogger(__name__)


class BestAverageWorstSizeTreeStatistics(Statistics):
    def __init__(self, format_string=None):
        if format_string is None:
            format_string = "best fitness {}\nworst fitness {}\naverage fitness {}\naverage size {}\n"
        super().__init__(format_string)

    def write_statistics(self, sender, data_dict):
        logger.info(f'generation #{data_dict["generation_num"]}')
        for index, sub_pop in enumerate(
            data_dict["population"].sub_populations
        ):
            logger.info(f"subpopulation #{index}")
            best_individual = sub_pop.get_best_individual()
            logger.info(
                self.format_string.format(
                    best_individual.get_pure_fitness(),
                    sub_pop.get_worst_individual().get_pure_fitness(),
                    sub_pop.get_average_fitness(),
                    np.average([ind.size() for ind in sub_pop.individuals]),
                ),
            )

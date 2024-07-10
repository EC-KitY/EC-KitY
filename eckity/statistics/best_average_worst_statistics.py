from sys import stdout
import logging

from eckity.statistics.statistics import Statistics

logger = logging.getLogger(__name__)


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

    """

    def __init__(self, format_string=None):
        if format_string is None:
            format_string = (
                "best fitness {}\nworst fitness {}\naverage fitness {}\n"
            )
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
                )
            )

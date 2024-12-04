import logging

from eckity.statistics.statistics import Statistics

logger = logging.getLogger(__name__)


class MinimalPrintStatistics(Statistics):
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

    # TODO tostring to indiv

    # Necessary for valid pickling, since modules cannot be pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    # Necessary for valid unpickling, since modules cannot be pickled
    def __setstate__(self, state):
        self.__dict__.update(state)

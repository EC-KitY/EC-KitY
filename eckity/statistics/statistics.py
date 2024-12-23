from abc import abstractmethod, ABC


class Statistics(ABC):
    """
    Abstract Statistics class.
    Provides statistics about the current evolution state.

    Parameters
    ----------
    format_string: str
        String format of the data to output.
        Value depends on the information the statistics provides.
        For more information, check out the concrete classes who extend this class.

    """

    def __init__(self, format_string):
        self.format_string = format_string

    @abstractmethod
    def write_statistics(self, sender, data_dict):
        """
        Write the statistics information using the format string field.

        Parameters
        ----------
        sender: object
            The object that this statistics provides information about.
            This class registers to a certain event that the sender object publishes.
            The statistics are shown as a callback to the event publication.
            For example, we can register a concrete Statistics sub-class to provide statistics after every generation
            of a concrete Algorithm sub-class.

        data_dict: dict(str, object)
            Relevant data to the statistics. Used to gain and provide information from the sender.

        Returns
        -------
        None.
        """
        pass

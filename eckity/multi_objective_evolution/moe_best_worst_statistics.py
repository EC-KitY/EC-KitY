from sys import stdout

from eckity.statistics.statistics import Statistics


class MOEBestWorstStatistics(Statistics):
    """
    Concrete Statistics class.
    Intended for Multi Objective Evolution.
    Provides statistics about the best and worst fitness fronts of every sub-population in
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
            format_string = "first front has {} individuals ({:.1f}%)\nfirst front's corners: {}\nlast front's corners: {}\n"
        super().__init__(format_string)
        self.output_stream = output_stream

    def write_statistics(self, sender, data_dict):
        print(
            f'generation #{data_dict["generation_num"]}',
            file=self.output_stream,
        )
        for index, sub_pop in enumerate(
            data_dict["population"].sub_populations
        ):
            first_front_corners = self.get_corners(sub_pop, 1)
            last_rank = sorted(
                list(
                    set(
                        [
                            indiv.fitness.front_rank
                            for indiv in sub_pop.individuals
                        ]
                    )
                )
            )[-1]
            last_front_corners = self.get_corners(sub_pop, last_rank)
            first_front = [
                indiv.get_pure_fitness()
                for indiv in sub_pop.individuals
                if indiv.fitness.front_rank == 1
            ]
            print(
                f"subpopulation #{index} has {last_rank} fronts",
                file=self.output_stream,
            )
            print(
                self.format_string.format(
                    len(first_front),
                    len(first_front) / len(sub_pop.individuals) * 100,
                    first_front_corners,
                    last_front_corners,
                ),
                file=self.output_stream,
            )

    def get_corners(self, sub_pop, rank):
        result = [
            indiv.get_pure_fitness()
            for indiv in sub_pop.individuals
            if indiv.fitness.front_rank == rank
            and indiv.fitness.crowding == float("inf")
        ]
        result = [list(x) for x in set(tuple(x) for x in result)]  # unique
        assert len(result) in [1, 2]
        return result

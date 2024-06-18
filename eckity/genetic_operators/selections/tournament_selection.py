import random
from overrides import override

from eckity.genetic_operators.selections.selection_method import (
    SelectionMethod,
)


class TournamentSelection(SelectionMethod):
    def __init__(
        self,
        tournament_size,
        higher_is_better=False,
        unique=False,
        events=None,
    ):
        """
        Tournament Selection method.
        In this method, small groups (known as tournaments) are created from
        randomly chosen individuals. The individual(s) with the best fitness
        scores are selected to reproduce the next generation.

        Parameters
        ----------
        tournament_size : int
            Size of each tournament.
            larger tournaments focus on exploitation,
            while small tournaments focus on exploration.
        higher_is_better : bool, optional
            is higher fitness better or worse, by default False
        unique : bool, optional
            whether tournaments can contain multiple copies of the same
            individual, by default False
        events : List[str], optional
            selection events, by default None
        """
        super().__init__(events=events, higher_is_better=higher_is_better)
        self.tournament_size = tournament_size
        self.unique = unique

    @override
    def select(self, source_inds, dest_inds):
        """
        The selection should add len(source_inds) individuals to dest_inds,
        so the required number of tournaments is the size of source
        individuals divided by the number of winners per tournament.
        `n_tournaments = len(source_inds) // self.operator_arity`
        """
        n_tournaments = (len(source_inds) - len(dest_inds)) // self.arity

        """
        Select the appropriate tournament creation function.
        `random.sample` selects k unique elements, and
        `random.choices` selects k elements with replacements
        """
        sel_func = random.sample if self.unique else random.choices

        # create all tournaments beforehand
        tournaments = [
            sel_func(source_inds, k=self.tournament_size)
            for _ in range(n_tournaments)
        ]

        # pick the winner of each tournament and add all winners to dest_inds
        winners = [self._pick_tournament_winner(tour) for tour in tournaments]
        dest_inds.extend(winners)

        self.selected_individuals = dest_inds

        return dest_inds

    def _pick_tournament_winner(self, tournament):
        winner = tournament[0]
        for participant in tournament[1:]:
            if participant.better_than(winner):
                winner = participant
        result = winner.clone()
        result.selected_by.append(type(self).__name__)
        return result

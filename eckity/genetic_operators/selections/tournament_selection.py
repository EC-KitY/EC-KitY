
from random import choices

from eckity.genetic_operators.selections.selection_method import SelectionMethod


class TournamentSelection(SelectionMethod):
    def __init__(self, tournament_size, higher_is_better=False, events=None):
        super().__init__(events=events, higher_is_better=higher_is_better)
        self.tournament_size = tournament_size

    def select(self, source_inds, dest_inds):
        # the selection should add len(source_inds) individuals to dest_inds, so the required number of tournaments
        # is the size of source individuals divided by the number of winners per tournament
        # n_tournaments = len(source_inds) // self.operator_arity
        n_tournaments = (len(source_inds) - len(dest_inds)) // self.arity

        # create all tournaments beforehand
        tournaments = [choices(source_inds, k=self.tournament_size) for _ in range(n_tournaments)]

        # pick the winner of each tournament and add all winners to dest_inds
        winners = [self._pick_tournament_winner(tour) for tour in tournaments]
        dest_inds.extend(winners)

        self.selected_individuals = dest_inds
        # self.publish("after_selection")   # TODO this already publishes 'after_operator' event

        return dest_inds

    def _pick_tournament_winner(self, tournament):
        winner = tournament[0]
        for participant in tournament[1:]:
            if participant.fitness.better_than(participant, winner.fitness, winner):
                winner = participant
        return winner.clone()

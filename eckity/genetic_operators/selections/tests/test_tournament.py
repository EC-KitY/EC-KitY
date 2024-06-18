import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_operators.selections.tournament_selection import (
    TournamentSelection,
)
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector


class TestTournament:
    @pytest.fixture(scope="class")
    def inds(self):
        return [
            BitStringVector(SimpleFitness(i),
                            length=0)
            for i in range(10)
        ]

    def test_with_replacement(self, inds):
        tournament = TournamentSelection(
            tournament_size=30, higher_is_better=True, replace=True
        )

        selected = tournament.select(inds[:2], [])
        first_winner = selected[0]
        assert first_winner.selected_by == [type(tournament).__name__]
        assert first_winner.cloned_from[0] == inds[0].id

    def test_without_replacement(self, inds):
        tournament = TournamentSelection(
            tournament_size=len(inds), higher_is_better=False, replace=False
        )

        selected = tournament.select(inds, [])
        first_winner = selected[0]
        assert first_winner.selected_by == [type(tournament).__name__]
        assert first_winner.cloned_from[0] == inds[0].id

    def test_tournament_too_big(self, inds):
        tournament = TournamentSelection(
            tournament_size=len(inds) + 1,
            higher_is_better=False,
            replace=False,
        )

        with pytest.raises(ValueError) as err_info:
            tournament.select(inds, [])
        
        assert "tournament size" in str(err_info).lower()

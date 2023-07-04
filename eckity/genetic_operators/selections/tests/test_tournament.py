import pytest

from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.genetic_encodings.ga.bit_string_vector import BitStringVector

class TestTournament:
	def test_selected_by(self):
		tournament = TournamentSelection(100, higher_is_better=False)
		inds = [
			BitStringVector(SimpleFitness(1, higher_is_better=False), length=4),
			BitStringVector(SimpleFitness(2, higher_is_better=False), length=4)
        ]
		selected = tournament.select(inds, [])
		winner = selected[0]
		assert winner.selected_by == ['TournamentSelection']
		assert winner.cloned_from == [inds[0].id]

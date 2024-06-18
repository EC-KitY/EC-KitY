from eckity.fitness.simple_fitness import SimpleFitness
from eckity.genetic_encodings.ga import BitStringVector
from eckity.genetic_operators import ElitismSelection


class TestElitism:
    def test_selected_by(self):
        elitism = ElitismSelection(2, higher_is_better=True)
        inds = [
            BitStringVector(SimpleFitness(1), length=4),
            BitStringVector(SimpleFitness(2), length=4),
        ]
        selected = elitism.select(inds, [])
        chosen = selected[0]
        assert chosen.selected_by == [type(elitism).__name__]
        assert chosen.cloned_from == [inds[1].id]

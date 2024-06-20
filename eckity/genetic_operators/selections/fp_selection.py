import numpy as np

from overrides import override
from scipy.special import softmax

from eckity.genetic_operators import SelectionMethod


class FitnessProportionateSelection(SelectionMethod):
    def __init__(
        self,
        higher_is_better=False,
        events=None,
    ):
        """
        Fitness Proportionate Selection, AKA Roulette Wheel Selection.
        In this method, the probability of selecting an individual from the
        population is proportional to its fitness score.
        Individuals with higher fitness have a higher chance of being
        selected as parents. This selection method simulates a roulette wheel,
        where the size of the slice for each individual on the wheel is
        determined by its fitness score

        Parameters
        ----------
        higher_is_better : bool, optional
            is higher fitness better or worse, by default False
        events : List[str], optional
            selection events, by default None
        """
        super().__init__(events=events, higher_is_better=higher_is_better)

    @override
    def select(self, source_inds, dest_inds):
        n_selected = len(source_inds) - len(dest_inds)

        fitness_scores = np.array(
            [ind.get_augmented_fitness() for ind in source_inds]
        )

        min_val = np.min(fitness_scores)

        if min_val < 0:
            raise ValueError(
                "Fitness scores must be non-negative for FP Selection"
            )

        # convert higher fitness scores to be better
        if not self.higher_is_better:
            # add smoothing (if necessary) to avoid division by zero
            smoothing = 1 if min_val == 0 else 0
            fitness_scores = 1 / (fitness_scores + smoothing)

        # generate a distribution of fitness scores
        fit_p = (
            softmax(fitness_scores)
            if np.sum(fitness_scores) != 1
            else fitness_scores
        )

        # select individuals proportionate to fitness
        selected_inds = np.random.choice(
            source_inds, size=n_selected, replace=True, p=fit_p
        )

        for selected_ind in selected_inds:
            clone = selected_ind.clone()
            clone.selected_by.append(type(self).__name__)
            dest_inds.append(clone)

        self.selected_individuals = dest_inds

        return dest_inds

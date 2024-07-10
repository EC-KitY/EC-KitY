from eckity.breeders.breeder import Breeder
from eckity.genetic_operators.selections.elitism_selection import (
    ElitismSelection,
)


class SimpleBreeder(Breeder):
    """
    A Simple version of Breeder class.
    All simple classes assume there is only one sub-population in population.
    """

    def __init__(self, events=None):
        super().__init__(events=events)
        self.selected_individuals = (
            []
        )  # TODO why do we need this field? what about applied_individuals?
        self.best_of_run = []  # TODO this field isn't used

    def apply_breed(self, population):
        """
        Apply elitism, selection and operator sequence on the sub-populations.
        In simple case, the operator sequence is applied on one sub-population.

        Parameters
        ----------
        population:
                Population of sub-populations of individuals.
                The operators will be applied on those individuals.

        Returns
        -------
        None.
        """
        for subpopulation in population.sub_populations:
            # Assert that operator arities are compatible with pop size
            # tuples of (selection, probability)
            selection_methods = subpopulation.get_selection_methods()
            selection_methods = [t[0] for t in selection_methods]

            operators_sequence = subpopulation.get_operators_sequence()
            operators = selection_methods + operators_sequence

            for oper in operators:
                if subpopulation.population_size % oper.arity != 0:
                    raise ValueError(
                        f"Operator {oper} arity must be "
                        f"dividable by population size"
                    )

            nextgen_population = []

            num_elites = subpopulation.n_elite
            if num_elites > 0:
                elitism_sel = ElitismSelection(
                    num_elites=num_elites,
                    higher_is_better=subpopulation.higher_is_better,
                )
                elitism_sel.apply_operator(
                    (subpopulation.individuals, nextgen_population)
                )

            self.selected_individuals = subpopulation.get_selection_methods()[
                0
            ][0].select(subpopulation.individuals, nextgen_population)

            # then runs all operators on next_gen
            nextgen_population = self._apply_operators(
                subpopulation.get_operators_sequence(),
                self.selected_individuals,
            )

            subpopulation.individuals = nextgen_population

    def _apply_operators(self, operator_seq, individuals_to_apply_on):
        """
        Apply a given operator sequence on a given list of individuals.
        The operators are done sequentially.

        Parameters
        ----------
        operator_seq: list of operators
                Operator sequence. Applied sequentially on the individuals.
        individuals_to_apply_on: list of individuals
                The individuals to apply the operator sequence on.

        Returns
        -------
        list of individuals
                The individuals list after the operators were applied on them.
        """
        for operator in operator_seq:
            operator_arity = operator.get_operator_arity()
            for i in range(0, len(individuals_to_apply_on), operator_arity):
                op_res = operator.apply_operator(
                    individuals_to_apply_on[i: i + operator_arity]
                )
                individuals_to_apply_on[i: i + operator_arity] = op_res
        return individuals_to_apply_on

    def event_name_to_data(self, event_name):
        if event_name == "after_selection":
            return {
                "selected_individuals": self.selected_individuals,
                "best_of_run": self.best_of_run,
            }

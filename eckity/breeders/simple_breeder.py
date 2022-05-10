from eckity.breeders.breeder import Breeder
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection


class SimpleBreeder(Breeder):
    def __init__(self,
                 events=None):
        super().__init__(events=events)
        self.selected_individuals = []  # TODO why do we need this field? what about applied_individuals?
        self.best_of_run = []   # TODO this field isn't used

    def apply_breed(self, population):
        """
        Apply elitism, selection method and the sub-population's operator sequence on each sub-population.
        In simple case, the operator sequence is applied on the one and only sub-population.

        Parameters
        ----------
        population:
            Population of sub-populations of individuals. The operators will be applied on those individuals.

        Returns
        -------
        None.
        """
        for subpopulation in population.sub_populations:
            nextgen_population = []

            num_elites = subpopulation.n_elite
            if num_elites > 0:
                elitism_sel = ElitismSelection(num_elites=num_elites, higher_is_better=subpopulation.higher_is_better)
                elitism_sel.apply_operator((subpopulation.individuals, nextgen_population))

            self.selected_individuals = subpopulation.get_selection_methods()[0][0] \
                .select(subpopulation.individuals, nextgen_population)

            # then runs all operators on next_gen
            nextgen_population = self._apply_operators(subpopulation.get_operators_sequence(),
                                                       self.selected_individuals)
            # TODO assert simple operators the has %0 with pop size

            subpopulation.individuals = nextgen_population

    def _apply_operators(self, operator_seq, individuals_to_apply_on):
        """
        Apply a given operator sequence on a given list of individuals.
        The operators are done sequentially.

        Parameters
        ----------
        operator_seq: list of operators
            Operator sequence. The operators will be applied sequentially on the given individuals.
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
                individuals_to_apply_on[i:i + operator_arity] = \
                    operator.apply_operator(individuals_to_apply_on[i:i + operator_arity])
        return individuals_to_apply_on

    def event_name_to_data(self, event_name):
        if event_name == "after_selection":
            return {"selected_individuals": self.selected_individuals,
                    "best_of_run": self.best_of_run}

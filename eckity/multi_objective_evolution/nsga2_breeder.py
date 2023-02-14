from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.elitism_selection import ElitismSelection


class NSGA2Breeder(SimpleBreeder):
	def __init__(self,
				 events=None):
		super().__init__(events=events)
		self.selected_individuals = []  # TODO why do we need this field? what about applied_individuals?

	def apply_breed(self, population):
		"""
        Apply elitism, selection method and the sub-population's operator sequence on each sub-population.
        In simple case, the operator sequence is applied on the one and only sub-population.

        adds the current generation to the next generation

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

			nextgen_population = self._create_next_gen(subpopulation)

			self.selected_individuals = subpopulation.get_selection_methods()[0][0] \
				.select(subpopulation.individuals, nextgen_population)

			subpopulation.individuals = nextgen_population

	def _create_next_gen(self, subpopulation):
		# oldgen_population = deepcopy(subpopulation.individuals)  # needed since apply operator changes the values of
		oldgen_population = [ind.clone() for ind in subpopulation.individuals]

		nextgen_population = self._apply_operators(subpopulation.get_operators_sequence(),
												   subpopulation.individuals)  # self.selected_individuals)

		oldgen_population += nextgen_population
		nextgen_population = oldgen_population

		for ind in nextgen_population:
			ind.fitness.set_not_evaluated()

		return nextgen_population

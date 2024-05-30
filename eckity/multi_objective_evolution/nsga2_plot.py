import matplotlib.pyplot as plt


class NSGA2Plot:
	def __init__(self, objective1='Objective function 1', objective2='Objective function 2', savefile=None):
		self.savefile = savefile
		self.objective1 = objective1
		self.objective2 = objective2

	def print_plots(self, sender, data_dict):
		population = data_dict["population"]
		ind = population.sub_populations[0].individuals[0]
		if len(ind.fitness.fitness) == 2:
			self._print_plots_2d(population)

	def _print_plots_2d(self, population):
		for sub_pop in population.sub_populations:
			fit0_list = [ind.fitness.fitness[0] for ind in sub_pop.individuals]
			fit1_list = [ind.fitness.fitness[1] for ind in sub_pop.individuals]
			self._print_plot(fit0_list, fit1_list)

	def _print_plot(self, ls1, ls2):
		plt.scatter(ls1, ls2)
		plt.xlabel(self.objective1)
		plt.ylabel(self.objective2)
		if self.savefile is not None:
			plt.savefig(self.savefile)
		plt.show()

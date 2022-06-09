class VectorOnePointMutation(ProbabilisticConditionOperator):
    def __init__(self, probability=1, arity=1, mutated_value_getter=None, events=None):
        super().__init__(probability=probability, arity=arity, events=events)
        if mutated_value_getter is None:
            mutated_value_getter = lambda x,y: y
        self.mutated_value_getter = mutated_value_getter

    def apply(self, individuals):
        """
        Perform ephemeral random constant (erc) mutation: select an erc node at random
        and add Gaussian noise to it.

        Returns
        -------
        None.
        """

        for j in range(len(individuals)):
            # selecting the point
            m_point = choice(individuals[j].get_vector())

            # getting the mutated value
            new_val = self.mutated_value_getter(j,m_point)

            # setting the mutated value
            individuals[j].set_cell_value(m_point, new_val)
        self.applied_individuals = individuals
        return individuals

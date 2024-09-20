from multiparent_dnc import NeuralCrossover
import torch


class NeuralCrossoverWrapper:
    def __init__(self, embedding_dim, sequence_length, num_embeddings, get_fitness_function, running_mean_decay=0.99,
                 batch_size=32, load_weights_path=None, freeze_weights=False, learning_rate=1e-3, epsilon_greedy=0.1,
                 use_scheduler=False, use_device='cpu', adam_decay=0, clip_grads=False, n_parents=2):
        self.device = use_device
        self.neural_crossover = NeuralCrossover(embedding_dim, embedding_dim, num_embeddings, sequence_length,
                                                n_parents=n_parents, device=use_device).to(
            self.device)
        self.running_mean_decay = running_mean_decay
        self.optimizer = torch.optim.Adam(self.neural_crossover.parameters(), lr=learning_rate, weight_decay=adam_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, verbose=True)
        self.get_fitness_function = get_fitness_function
        self.batch_size = batch_size
        self.n_parents = n_parents
        self.batch_stack_fitness_values = []
        self.sampled_action_space = []
        self.sampled_solutions = []
        self.load_weights_path = load_weights_path
        self.freeze_weights = freeze_weights
        self.epsilon_greedy = epsilon_greedy
        self.use_scheduler = use_scheduler
        self.clip_grads = clip_grads
        self.acc_batch_length = 0

        if self.load_weights_path is not None:
            self.neural_crossover.load_state_dict(torch.load(self.load_weights_path))

    def get_batch_and_clear(self):
        """
        Returns the batch of parents and fitness values and clears the batch.
        """
        fitness_values = torch.cat(self.batch_stack_fitness_values, dim=0).unsqueeze(1).to(self.device)
        sampled_action_space = torch.cat(self.sampled_action_space, dim=0).to(self.device)
        sampled_solutions = torch.cat(self.sampled_solutions, dim=0).to(self.device)

        self.clear_stacks()

        return fitness_values, sampled_action_space, sampled_solutions

    def clear_stacks(self):
        """
        Clears the batch stacks.
        """
        self.batch_stack_fitness_values.clear()
        self.sampled_action_space.clear()
        self.sampled_solutions.clear()

    def run_epoch(self):
        """
        Performs one step of training on the neural crossover.
        """
        if self.freeze_weights:
            self.clear_stacks()
            return

        total_batches_length = self.acc_batch_length
        if total_batches_length < self.batch_size:
            return

        self.acc_batch_length = 0

        fitness_values, sampled_action_space, sampled_solutions = self.get_batch_and_clear()
        self.optimizer.zero_grad()
        sampled_solutions_proba = torch.gather(sampled_action_space, 2, sampled_solutions.unsqueeze(2)).squeeze(-1).to(
            self.device)
        loss = -torch.mean(
            torch.log(sampled_solutions_proba) * (fitness_values.type(torch.DoubleTensor)).to(self.device))

        loss.backward()

        if self.clip_grads:
            torch.nn.utils.clip_grad_norm_(self.neural_crossover.parameters(), 1.0)

        self.optimizer.step()

        if self.use_scheduler:
            self.scheduler.step(loss)

        print(f'loss: {loss}, reward: {torch.mean(fitness_values.type(torch.DoubleTensor))}')

    def combine_parents_uniform(self, parents_matrix):
        """
        Uses the neural crossover to select the crossover points from the parents.
        """
        if self.freeze_weights:
            self.neural_crossover.eval()

        parents_matrix = parents_matrix.to(self.device)

        attention_values, selected_crossovers_indices = self.neural_crossover(parents_matrix,
                                                                              epsilon_greedy=self.epsilon_greedy)
        self.sampled_action_space.append(attention_values)
        self.sampled_solutions.append(selected_crossovers_indices)
        return torch.gather(parents_matrix.permute(1, 2, 0), dim=2,
                            index=selected_crossovers_indices.unsqueeze(-1)).squeeze(-1)

    def update_batch_stack(self, fitness_values):
        """
        Updates the batch stack.
        """
        self.batch_stack_fitness_values.append(fitness_values)

    def get_crossover(self, parents_matrix):
        """
        Uses the neural crossover to select the crossover points from the parents.
        Then performs one step of training on the neural crossover.
        :param parents_matrix: parents to crossover
        :return: resulting crossover individuals
        """
        parents_matrix = torch.Tensor(parents_matrix).type(torch.LongTensor)

        selected_crossover_func = self.combine_parents_uniform

        child1 = selected_crossover_func(parents_matrix)
        child2 = selected_crossover_func(parents_matrix)

        child1_fitness_values = [self.get_fitness_function(child) for child in
                                 child1.detach().cpu().numpy()]

        child2_fitness_values = [self.get_fitness_function(child) for child in
                                 child2.detach().cpu().numpy()]

        child1_fitness_values = torch.Tensor(child1_fitness_values).type(torch.FloatTensor)
        child2_fitness_values = torch.Tensor(child2_fitness_values).type(torch.FloatTensor)

        self.update_batch_stack(child1_fitness_values)
        self.update_batch_stack(child2_fitness_values)
        self.run_epoch()

        return child1.detach().cpu().numpy().tolist(), child2.detach().cpu().numpy()

    def cross_pairs(self, parents_pairs):
        if len(parents_pairs) == 0:
            return []

        parents_grouped = list(zip(*parents_pairs))
        parents_matrix = torch.cat([torch.unsqueeze(torch.tensor(group), 0) for group in parents_grouped],
                                   dim=0)
        self.acc_batch_length += parents_matrix.shape[1]
        child1, child2 = self.get_crossover(parents_matrix)
        return list(zip(child1, child2))

    def save_weights(self, path):
        torch.save(self.neural_crossover.state_dict(), path)

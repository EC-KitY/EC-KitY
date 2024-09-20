import torch
from sequence_linear_embedding import SequenceLinearEmbedding
from pointer_encoder import PointerEncoder
from pointer_decoder import PointerDecoder


class NeuralCrossover(torch.nn.Module):
    def __init__(self, input_size, hidden_size, n_embeddings, ind_length, num_layers=1, dropout=0, n_parents=2,
                 device='cuda'):
        super(NeuralCrossover, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_parents = n_parents
        self.encoder = PointerEncoder(input_size, hidden_size, num_layers, dropout)
        self.decoder = PointerDecoder(input_size, self.n_parents * hidden_size, num_layers, dropout)
        self.embedding = SequenceLinearEmbedding(n_embeddings, input_size)
        self.initial_trainable_input = torch.nn.Parameter(torch.randn(1, input_size))
        self.ind_length = ind_length
        self.device = device

    def forward(self, parents_matrix, epsilon_greedy=0.1):
        """
        :param how_many_points: how many points to perform the crossover on (none means uniform crossover)
        :param epsilon_greedy: (float) probability of choosing a random action
        :param parents_matrix: (n_parents, batch_size, seq_len)
        :return: (batch_size, seq_len)
        """
        batch_size = parents_matrix.shape[1]

        parents_embeddings = self.embedding(parents_matrix)
        # parents_embeddings: (n_parents, batch_size, seq_len, input_size)

        # h: (num_layers, batch_size, hidden_size)
        # c: (num_layers, batch_size, hidden_size)
        collected_hidden_states = []
        collected_cell_states = []
        for parent_index in range(parents_embeddings.shape[0]):
            _, (hidden_parent_state, cell_parent_state) = self.encoder(parents_embeddings[parent_index])
            collected_cell_states.append(cell_parent_state)
            collected_hidden_states.append(hidden_parent_state)

        h = torch.cat(collected_hidden_states, dim=2)
        c = torch.cat(collected_cell_states, dim=2)

        # y: (batch_size, seq_len)
        attention_values = []
        distributions_samples = []
        previous_chosen_sequence_repeat_for_batch = self.initial_trainable_input.repeat(batch_size, 1, 1)

        for i in range(self.ind_length):
            # a: (batch_size, 1)
            # h: (num_layers, batch_size, hidden_size)
            # c: (num_layers, batch_size, hidden_size)

            current_embeddings = [
                parents_embeddings[parent_index][:, i:i + 1, :]
                for parent_index in range(parents_embeddings.shape[0])
            ]
            ref_between_parents = torch.cat(current_embeddings, dim=1)

            a, (h, c) = self.decoder(previous_chosen_sequence_repeat_for_batch, ref_between_parents, (h, c))
            # y: (batch_size, 1)

            if torch.rand(1) < epsilon_greedy:
                sample_from_a_dist = torch.flatten(torch.randint(0, a.shape[1], (batch_size, 1))).to(self.device)
            else:
                sample_from_a_dist = torch.distributions.Categorical(a).sample().to(self.device)

            distributions_samples.append(sample_from_a_dist)
            attention_values.append(a)
            # previous_chosen_sequence_repeat_for_batch: (batch_size, 1, input_size)
            previous_chosen_sequence_repeat_for_batch = self.embedding(sample_from_a_dist).unsqueeze(1)

        # y: (batch_size, seq_len)
        attention_values = torch.stack(attention_values, dim=1)
        distributions_samples = torch.stack(distributions_samples, dim=1)
        return attention_values, distributions_samples

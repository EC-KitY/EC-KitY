import torch


class SequenceLinearEmbedding(torch.nn.Module):
    """
    Embed a sequence of integers into a sequence of vectors.
    """

    def __init__(self, num_embeddings, embedding_dim):
        super(SequenceLinearEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x):
        # x: (batch_size, seq_len)
        # y: (batch_size, seq_len, embedding_dim)
        y = self.embedding(x)
        return y

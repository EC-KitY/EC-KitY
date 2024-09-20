import torch


class Attention(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Attention, self).__init__()
        self.w_ref = torch.nn.Linear(input_size, hidden_size, bias=False)
        self.w_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = torch.nn.Linear(hidden_size, 1, bias=False)

        
    def forward(self, ref, q):
        """
        :param ref: (batch_size, seq_len, input_size)
        :param q: (batch_size, hidden_size)
        :return: (batch_size, seq_len)
        """
        # ref: (batch_size, seq_len, hidden_size)
        ref = self.w_ref(ref)
        # q: (batch_size, 1, hidden_size)
        q = self.w_q(q).unsqueeze(1)
        # e: (batch_size, seq_len, hidden_size)
        e = torch.tanh(ref + q)
        # e: (batch_size, seq_len)
        e = self.v(e).squeeze(-1)
        # a: (batch_size, seq_len)
        a = torch.softmax(e, dim=1)
        return a

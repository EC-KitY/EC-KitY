import torch
from attention import Attention


class PointerDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(PointerDecoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = Attention(input_size, hidden_size)
        self.w = torch.nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x, ref, previous_hidden):
        """
        :param x: (batch_size, 1, input_size)
        :param ref: (batch_size, seq_len, input_size)
        :param previous_hidden: (num_layers, batch_size, hidden_size)
        :return: (batch_size, 1)
        """
        # h: (num_layers, batch_size, hidden_size)
        # c: (num_layers, batch_size, hidden_size)
        lstm_out, (lstm_hidden_h, lstm_hidden_c) = self.lstm(x, previous_hidden)
        # a: (batch_size, seq_len)
        a = self.attention(ref, lstm_hidden_h[-1])
        return a, (lstm_hidden_h, lstm_hidden_c)
        
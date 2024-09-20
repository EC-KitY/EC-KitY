import torch


class PointerEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(PointerEncoder, self).__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x):
        """
        Uses the LSTM to encode the input sequence into a single vector.
        :return:
        """
        # x: (batch_size, seq_len, input_size)
        # h: (num_layers, batch_size, hidden_size)
        # c: (num_layers, batch_size, hidden_size)
        lstm_out, (lstm_hidden_h, lstm_hidden_c) = self.lstm(x)
        return lstm_out, (lstm_hidden_h, lstm_hidden_c)
        # return lstm_hidden_h[-1]

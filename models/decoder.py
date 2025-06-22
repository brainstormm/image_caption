import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    # x.shape : [batch_size, sequence_length, input_size]
    # hidden.shape : [num_layers,batch_size,hidden_size]
    def forward(self, x, hidden) -> tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.rnn(x, hidden)
        # output.shape : [batch_size, sequence_length, hidden_size]
        # hidden.shape : [num_layers, batch_size, hidden_size]

        # Apply linear layer to all time steps
        batch_size, seq_length, hidden_size = output.shape
        output = output.reshape(
            -1, hidden_size
        )  # (batch_size * seq_length, hidden_size)
        output = self.fc(output)  # (batch_size * seq_length, output_size)
        output = output.reshape(
            batch_size, seq_length, -1
        )  # (batch_size, seq_length, output_size)

        return output, hidden


def init_hidden(self, batch_size) -> torch.Tensor:
    return torch.zeros(self.num_layers, batch_size, self.hidden_size)

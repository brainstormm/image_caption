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

    def forward(self, x, hidden) -> tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.rnn(x, hidden)
        output = self.fc(output[:, -1, :])
        return output, hidden


def init_hidden(self, batch_size) -> torch.Tensor:
    return torch.zeros(self.num_layers, batch_size, self.hidden_size)

import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, input_size=4096, hidden_size=1024, num_layers=2):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        outputs, (hidden, cell) = self.lstm(x)
        return (hidden, cell)


class Decoder(nn.Module):
    def __init__(
        self, input_size=4096, hidden_size=1024, output_size=4096, num_layers=2
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # x: tensor of shape (batch_size, seq_length, hidden_size)
        output, (hidden, cell) = self.lstm(x, hidden)
        prediction = self.fc(output)
        return prediction, (hidden, cell)



class LSTMAE(nn.Module):
    """LSTM-based Auto Encoder"""

    def __init__(self, input_size, hidden_size, latent_size, device=torch.device("cuda")):
        """
        input_size: int, batch_size x sequence_length x input_dim
        hidden_size: int, output size of LSTM AE
        latent_size: int, latent z-layer size
        num_lstm_layer: int, number of layers in LSTM
        """
        super(LSTMAE, self).__init__()
        self.device = device

        # dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        # lstm ae
        self.lstm_enc = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
        )
        self.lstm_dec = Decoder(
            input_size=input_size,
            output_size=input_size,
            hidden_size=hidden_size,
        )

        self.criterion = nn.MSELoss()

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape

        enc_hidden = self.lstm_enc(x)

        temp_input = torch.zeros((batch_size, seq_len, feature_dim), dtype=torch.float).to(
            self.device
        )
        hidden = enc_hidden
        reconstruct_output, hidden = self.lstm_dec(temp_input, hidden)
        reconstruct_loss = self.criterion(reconstruct_output, x)

        return reconstruct_loss, reconstruct_output, (0, 0)
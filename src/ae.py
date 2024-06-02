import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import Tensor
from torchsummary import summary

from utils import reshape_matrix


class Encoder(nn.Module):
    def __init__(self, dims: list[int], act=nn.ReLU()):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            act,
            nn.Linear(dims[1], dims[2]),
            act,
            nn.Linear(dims[2], dims[3]),
            act,
        )

    def forward(self, x):
        x = x.reshape(-1, x.shape[0] * x.shape[1])
        x = self.net(x)
        return x


class Decoder(nn.Module):
    def __init__(self, dims: list[int], act=nn.ReLU(inplace=True)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dims[0], dims[1]),
            act,
            nn.Linear(dims[1], dims[2]),
            act,
            nn.Linear(dims[2], dims[3]),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class AE(nn.Module):

    def __init__(self, dims: list[int]):
        super().__init__()
        assert len(dims) == 4, "Too many sizes"

        self.encoder = Encoder(dims)
        self.decoder = Decoder(dims[::-1])
        self.loss = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def criterion(self, x, x_hat):
        reshaped = x.reshape(-1, x.shape[0] * x.shape[1])
        return self.loss(reshaped, x_hat)


if __name__ == "__main__":
    M = 100
    N = 100

    dims = [M * N, 1024, 512, 64]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(dims).to(device)

    data = torch.randn((M, N), dtype=torch.float32).to(device)
    print(data)

    model.train()
    for epoch in range(10):
        output = model(data)
        loss = model.criterion(data, output)
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

    output = model(data)
    print(data)

    # summary(model, (M, N))

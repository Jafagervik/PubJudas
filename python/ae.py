import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchsummary import summary
from torch import Tensor

from utils import reshape_matrix

class Encoder(nn.Module):
    def __init__(self, dims: list[int], act = nn.ReLU):
        super().__init__()
        assert len(dims) == 4, "Too many sizes"
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.act = act

    def forward(self, x):
        x = reshape_matrix(x)
        print(x.shape)
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.act(x)

        return x

class Decoder(nn.Module):
    def __init__(self, dims: list[int], act = nn.ReLU):
        super().__init__()
        assert len(dims) == 4, "Too many sizes"
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        self.act = act

    def forward(self, x):
        x = reshape_matrix(x)
        x = self.fc1(x)
        x = self.act(x)
        
        x = self.fc2(x)
        x = self.act(x)

        x = self.fc3(x)
        x = self.act(x)

        return x
    
class AE(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, dims: list[int]):
        super(AE, self).__init__()
        assert len(dims) == 4, "Too many sizes"

        self.encoder = Encoder(dims)
        self.decoder = Decoder(dims[::-1])

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    M = 100
    N = 100

    dims = [M*N, 1024, 512, 64]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AE(dims).to(device)

    summary(model, (M, N))

def old():
    input_size = M * N
    hidden_size = 1024
    num_layers = 4

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = torch.rand(M, N, dtype=torch.float32, device=device)

    #import seaborn as sns
    #sns.heatmap(img.detach().cpu(), cmap="coolwarm")
    #plt.show()

    l = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)


    print(img.size())
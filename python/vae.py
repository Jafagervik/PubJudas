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
    
class VAE(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, dims: list[int]):
        super(VAE, self).__init__()
        assert len(dims) == 4, "Too many sizes"

        self.encoder = Encoder(dims)
        self.decoder = Decoder(dims[::-1])
        
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def reparametrize(self, x):
        return x

    def elbo_loss(self, *args, **kwargs):
        recons = args[0]
        inp = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = 25e-5  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, inp)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss
        return {
            "loss": loss,
            "Reconstruction_Loss": recons_loss.detach(),
            "KLD": -kld_loss.detach(),
        }

if __name__ == "__main__":
    M = 100
    N = 100

    dims = [M*N, 1024, 512, 64]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = VAE(dims).to(device)

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
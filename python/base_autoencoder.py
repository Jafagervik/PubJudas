import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tqdm import tqdm
from abc import abstractmethod

class AbstractAutoEncoder(nn.Module):
    @abstractmethod
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x: Tensor):
        encoded = self.encoder(x)
        decoded = self.decoder(x)
        return decoded

    #def fit(self, mode='ae', train_data=None, num_epochs=10, bs=32, lr=1f-3, momentum=0., **kwargs):
    #    return fit_ae(mode=self, mode=mode, train_data=train_data, num_epochs=num_epochs, bs=bs, lr=lr, momentum=momentum, **kwargs)

    @abstractmethod 
    def criterion(self):
        pass

class AE(AbstractAutoEncoder):
    def __init__(self, input_dim: int = 50*50, use_bias: bool = True):
        super().__init__()
        self.type="AE"
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 100, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50, bias=use_bias),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Linear(50, 100, bias=use_bias),
            nn.ReLU(inplace=True),
            nn.Linear(100, input_dim, bias=use_bias),
            nn.Sigmoid(),
        )

if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.rand(5000, 5000, dtype=torch.float32)

    dataloader = torch.utils.DataLoader(data, batch_size=1, suffle=False)
    criterion = nn.MSELoss()

    model = AE().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    EPS = 5 
    loss = None

    for epoch in range(EPS):
        





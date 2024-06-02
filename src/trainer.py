import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset


class MockDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class Trainer:
    """
    Trainer class to train the model

    Args:
        model (nn.Module): Pytorch model
        train_loader (DataLoader): Pytorch dataloader
        optimizer (Optimizer): Pytorch optimizer
        device (torch.device): Pytorch device
        loss (nn.Module): Pytorch loss function
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        optimizer: Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.best_loss = float("inf")
        self.losses = []

    def _run_batch(self, source: Tensor, epoch: int):
        self.optimizer.zero_grad()

        outputs = self.model(source)
        # Loss is a dict
        loss = self.model.loss_function(outputs, source)
        self.losses.append(loss.item()["loss"])
        print(f"Loss: {loss.item()}")

        if loss.item() < self.best_loss:
            self.best_loss = loss.item()
            self._save_checkpoint(epoch)

        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch: int):
        print(f"Epoch {epoch + 1}")
        for source in self.train_loader:
            # source = source.to(self.device)
            self._run_batch(source, epoch)

    def train(self, epochs: int):
        print("Training...")
        self.model.train()
        for epoch in range(epochs):
            self._run_epoch(epoch)

    def _save_checkpoint(self, epoch: int, final: bool = False):
        # TODO: move model to cpu before saving
        # TODO: Only save if rank is 0
        ckp = self.model.state_dict()
        PATH = "final.pt" if final else "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch + 1} | Training checkpoint saved at {PATH}")

    def _load_from_checkpoint(self, path: str):
        ckp = torch.load(path)
        self.model.load_state_dict(ckp)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            for source in self.train_loader:
                # source = source.to(self.device)
                outputs = self.model(source)
                loss = self.model.loss(outputs, source)
                print(f"Test Loss: {loss.item()}")


if __name__ == "__main__":
    SEED = 1234
    M = 512
    EPOCHS = 20
    N = 10

    torch.manual_seed(SEED)
    random.seed(SEED)

    # Autoencoder model
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 3),
        nn.ReLU(),
        nn.Linear(3, 5),
        nn.ReLU(),
        nn.Linear(5, 10),
        nn.Sigmoid(),
    )

    data = [torch.randn(10) for _ in range(100)]
    dataset = MockDataset(data)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loader = DataLoader(dataset, batch_size=10, shuffle=False)
    device = torch.device("cpu")
    loss = nn.MSELoss()
    trainer = Trainer(model, train_loader, optimizer, device, loss)

    trainer.train(EPOCHS)

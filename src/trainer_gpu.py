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
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.best_loss = float("inf")
        self.losses = []


    def _run_batch(self, source: Tensor, epoch: int, **kwargs):
        self.optimizer.zero_grad()

        M, N = source.shape

        source = source.reshape(-1, M * N)

        outputs = self.model(source)
        # Loss is a list of 4 elems
        elbo_loss = self.model.loss_function(*outputs, **kwargs)

        self.losses.append(elbo_loss)

        if elbo_loss["loss"] < self.best_loss:
            self.best_loss = elbo_loss["loss"]
            print(f"{self.best_loss.item():.3f}")
            self._save_checkpoint(epoch)

        elbo_loss["loss"].backward()

    def _run_epoch(self, epoch: int, **kwargs):
        print(f"Epoch {epoch + 1}")
        for batch in self.train_loader:
            for data in batch:
                data = data.to(self.device)
                self._run_batch(data, epoch, **kwargs)
            self.optimizer.step()

    def train(self, epochs: int, **kwargs):
        print("Training...")
        self.model.train()
        for epoch in range(epochs):
            self._run_epoch(epoch, **kwargs)
        self._save_checkpoint(epochs, final=True)

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


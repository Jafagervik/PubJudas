import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from .data import PubDASDataset

import numpy as np

def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

class Engine:
    def __init__(self) -> None:
        self.model = F.linear(1, 12)   
        
    def _run_batch(self): 
        pass 

    def _run_epoch(self, epoch: int):
        pass

    def _save_checkpoint(self, epochs: int):
        for epoch in range(epochs):
            self._run_epoch(epoch)
    
    def train(self, epoch: int): 
        ckp = self.model.module.state_dict()
        PATH = "checkpoint.pt"
        torch.save(ckp, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

class AE(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(AE, self).__init__()

    def forward(self, x):
        return x

def load_train_objects(device):
    train_set = None
    model = AE().to(device)
    optimizer = torch.Adam(model.parameters(), lr=1e-2)
    return train_set, model, optimizer

def prepare_dataloader(dataset: Dataset, batch_size: int):
    """
        Distributred Datasampler for dataloader
    """
    return DataLoader(
        dataset,
        batch_size,
        pin_memory=True,
        shuffle=False, 
        sampler=DistributedSampler
    )


def main():
    pass 

if __name__ == "__main__":
    main()
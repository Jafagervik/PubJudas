import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from ae import AE
from data import PubDASDataset, prepare_dataloader
from engine import Engine
from plots import *
from utils import parse_args


# TODO: Not needed if we use torchrun
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


def load_train_objects(device):
    train_set = None
    model = AE(dims=[10, 20, 30]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    return train_set, model, optimizer


def main():
    pass


if __name__ == "__main__":
    parser = parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    main()

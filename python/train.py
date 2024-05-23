import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from data import PubDASDataset
from engine import Engine
from plots import *
from ae import AE

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


def load_train_objects(device):
    train_set = None
    model = AE().to(device)
    optimizer = torch.Adam(model.parameters(), lr=1e-2)
    return train_set, model, optimizer


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
    args = parser.parse_args()

    return args

def main():
    pass 

if __name__ == "__main__":
    parser = parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    main()
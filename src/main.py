import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from ae import AE
from data import PubDASDataset
from engine import Engine
from plots import *
from utils import parse_args


def main():
    pass


if __name__ == "__main__":
    parser = parse_args()
    world_size = torch.cuda.device_count()
    print(world_size)
    main()


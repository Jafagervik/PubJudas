import argparse
import os
import random as rnd

import torch
import yaml
from torch import Tensor


def reshape_matrix(d: Tensor):
    return d.view(-1).unsqueeze(0)


def reshape_to_matrix(data: Tensor, M: int, N: int):
    return data.reshape(M, N)


def get_devices():
    """Get device names."""
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))


def seed_all(seed: int):
    """Seed all random number generators."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    rnd.seed(seed)
    # os.environ["PYTHONHASHSEED"] = str(seed)


def get_cpu_and_gpu_count():
    """CPU and GPU count for mpispawn and torchrun"""
    return os.cpu_count(), torch.cuda.device_count()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch distributed training")

    parser.add_argument(
        "--config",
        "-c",
        dest="filename",
        metavar="FILE",
        help="path to the config file",
        #default="../configs/vae.yaml",
        default=os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'configs', "vae.yaml"))
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="12355")

    return parser.parse_args()


def get_config(path: str):
    with open(path, "r") as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

import argparse
import os

import torch
from torch import Tensor


def reshape_matrix(d: Tensor):
    return d.view(-1).unsqueeze(0)


def reshape_to_matrix(data: Tensor, M: int, N: int):
    return data.reshape(M, N)


def get_devices():
    """Get device names."""
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))


def get_cpu_and_gpu_count():
    """CPU and GPU count for mpispawn and torchrun"""
    return os.cpu_count(), torch.cuda.device_count()


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch distributed training")

    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--backend", type=str, default="nccl")
    parser.add_argument("--master_addr", type=str, default="127.0.0.1")
    parser.add_argument("--master_port", type=str, default="12355")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_model", type=bool, default=True)
    parser.add_argument("--save_path", type=str, default="model.pth")
    parser.add_argument("--load_model", type=bool, default=False)

    return parser.parse_args()


if __name__ == "__main__":
    a = torch.rand(4, 4, dtype=torch.float32)

    print(a, a.shape, a.size())

    a = a.flatten()

    print(a)
    print(a, a.shape, a.size())

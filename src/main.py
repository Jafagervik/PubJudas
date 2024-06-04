import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset

from models.vae import VAE

# from data import PubDASDataset
from plots import *
from trainer_gpu import MockDataset, Trainer
from utils import get_config, parse_args, seed_all

DEBUG = 0


def main():
    args = parse_args()
    config = get_config(args.filename)

    seed_all(config["exp_params"]["manual_seed"])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    print(f"Training on {device}")

    model = VAE(
        M=config["model_params"]["M"],
        N=config["model_params"]["N"],
        latent_dim=config["model_params"]["latent_dim"],
        hidden_dim=config["model_params"]["hidden_dim"],
    ).to(device)

    if DEBUG > 1:
        print(model)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["exp_params"]["lr"],
        weight_decay=config["exp_params"]["weight_decay"],
    )

    data = torch.randn(
        100,
        config["model_params"]["M"],
        config["model_params"]["N"],
        dtype=torch.float32,
    )

    if DEBUG > 1:
        print(data.shape)

    dataset = MockDataset(data)
    train_loader = DataLoader(
        dataset,
        batch_size=config["data_params"]["batch_size"],
    )

    trainer = Trainer(model, train_loader, optimizer, device)

    trainer.train(config["trainer_params"]["epochs"], **config)

    ls = [losses["loss"].cpu() for losses in trainer.losses]

    plt.plot(ls)
    plt.show()

    # TODO: Anomaly detection


if __name__ == "__main__":
    main()

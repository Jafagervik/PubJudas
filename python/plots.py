import matplotlib.pyplot as plt
import torch
from torch import Tensor
import numpy as np
import seaborn as sns

def plot_dasmap(data: Tensor, save: bool = False):
    A = torch.rand(10, 10, dtype=torch.float32)

    plt.figure(figsize = (10,8))
    sns.heatmap(A, cmap="coolwarm")
    if save:
        plt.savefig("dasmap.png")
    else:
        plt.show()
        

def plot_ae_losses(losses, save: bool = False):
    plt.plot(losses)
    if save:
        plt.savefig("ae_losses.png")
    else:
        plt.show()

def plot_vae_losses(losses: dict, save: bool = False):
    pass

def plot_anomalies(anomalies, save: bool = False):
    pass

import requests 

if __name__ == "__main__":
    data = torch.rand(10, 10, dtype=torch.float32)
    torch.rand
    plot_dasmap(data)
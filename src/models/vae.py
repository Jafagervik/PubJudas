import torch
import torch.nn as nn
import torch.nn.functional as F
from customtypes import Any, List, Tensor
from torchsummary import summary

from base_autoencoder import BaseVAE


class VAE(BaseVAE):
    def __init__(self, M: int, N: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.M = M
        self.N = N
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        self.rec_loss = nn.MSELoss()

        self.encoder = nn.Sequential(
            nn.Linear(M * N, hidden_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, M * N),
            nn.Sigmoid(),
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        x = input.reshape(-1, input.shape[0] * input.shape[1])

        result = self.encoder(x)
        mu, log_var = self.fc_mu(result), self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder(z)
        return result.reshape(self.M, self.N)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        Elbo = Reconstruction + KL Divergence
        """
        recons, input, mu, log_var = args

        kld_weight = kwargs["M_N"]  # Account for the minibatch samples from the dataset
        recons_loss = self.rec_loss(recons, input)

        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0
        )

        loss = recons_loss + kld_weight * kld_loss

        return {
            "loss": loss,
            "recloss": recons_loss.detach(),
            "kld": -kld_loss.detach(),
        }

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(batch_size, self.latent_dim).to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]

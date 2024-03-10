from typing import Tuple
import torch
from torch import nn
from torch.distributions.normal import Normal
from pytorch_lightning import LightningModule
from torchmetrics import MetricCollection

from preprocessing.neural_networks.model import PreprocessingModel


class ConvBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 2,
    ) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DeconvBlock(LightningModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        stride: int = 2,
        output_padding: int = 1,
    ) -> None:
        super(DeconvBlock, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.deconv(x)


class LatentSpaceSampler(LightningModule):
    def forward(
        self, mean_vector: torch.Tensor, log_variance_vector: torch.Tensor
    ) -> torch.Tensor:
        batch_size, latent_dim = mean_vector.shape
        epsilon_noise = (
            Normal(0, 1)
            .sample(torch.Size((batch_size, latent_dim)))
            .to(mean_vector.device)
        )
        # Use the reparameterization trick to sample a latent vector
        # from the distribution defined by the mean and log variance vectors
        return mean_vector + torch.exp(0.5 * log_variance_vector) * epsilon_noise


class Encoder(LightningModule):
    def __init__(
        self, image_shape: Tuple[int, int], embedding_dim: int, n_channels: int = 1
    ) -> None:
        super(Encoder, self).__init__()
        # Assuming image_size is the size of one side of a square image
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels=n_channels, out_channels=32),
            ConvBlock(in_channels=32, out_channels=64),
            ConvBlock(in_channels=64, out_channels=128),
        )
        self.flatten = nn.Flatten()
        conv_output_size = self._get_conv_output_size(image_shape)
        self.fc_mean = nn.Linear(conv_output_size, embedding_dim)
        self.fc_log_var = nn.Linear(conv_output_size, embedding_dim)
        self.sampling = LatentSpaceSampler()

    def _get_conv_output_size(self, image_shape: Tuple[int, int]) -> int:
        with torch.no_grad():
            dummy_input = torch.zeros(
                (1, 1, image_shape[0], image_shape[1]), dtype=torch.float32
            )
            output: torch.Tensor = self.conv_blocks(dummy_input)
            output_size = (
                output.numel() // output.shape[0]
            )  # Calculate total elements per item in batch
        return output_size

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.conv_blocks(x)
        x = self.flatten(x)
        z_mean = self.fc_mean(x)
        z_log_var = self.fc_log_var(x)
        z = self.sampling(z_mean, z_log_var)
        return z_mean, z_log_var, z


class Decoder(LightningModule):
    def __init__(
        self,
        embedding_dim: int,
        initial_size: int,
        initial_channels: int,
        output_channels: int = 1,
    ) -> None:
        super(Decoder, self).__init__()
        self.initial_channels = initial_channels
        self.initial_size = initial_size
        self.embedding_dim = embedding_dim
        self.fc = nn.Linear(
            embedding_dim, self.initial_channels * initial_size * initial_size
        )

        self.deconv_blocks = nn.Sequential(
            DeconvBlock(in_channels=self.initial_channels, out_channels=64),
            DeconvBlock(in_channels=64, out_channels=32),
            DeconvBlock(in_channels=32, out_channels=output_channels, output_padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = x.view(
            -1, self.initial_channels, self.initial_size, self.initial_size
        )  # Reshape to start deconvolutions
        x = self.deconv_blocks(x)
        return torch.sigmoid(x)


class VariationalAutoencoder(PreprocessingModel):
    def __init__(
        self,
        image_shape: Tuple[int, int],
        embedding_dim: int,
        n_channels: int = 1,
        initial_size: int = 32,
        initial_channels: int = 128,
        learning_rate: float = 1e-4,
        metrics: MetricCollection | None = None,
    ) -> None:
        super(VariationalAutoencoder, self).__init__(
            learning_rate=learning_rate, metrics=metrics
        )
        self.encoder = Encoder(
            n_channels=n_channels, image_shape=image_shape, embedding_dim=embedding_dim
        )
        self.decoder = Decoder(
            embedding_dim=embedding_dim,
            output_channels=n_channels,
            initial_size=initial_size,
            initial_channels=initial_channels,
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_mean, z_log_var, z = self.encoder(x)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

import torch
from diffusers import UNet2DModel
from torch import nn


class ConditionalUnet(nn.Module):
    """
    A conditional U-Net model that combines an input sample and conditioning information
    by concatenating them along the channel dimension,
    and processes them using a U-Net architecture from the `diffusers` library.

    Args:
        params_dict (dict): A dictionary of parameters to initialize the `UNet2DModel`.

    Attributes:
        model (UNet2DModel): The U-Net model for denoising or diffusion tasks.
    """

    def __init__(self, params_dict):
        super().__init__()

        self.model = UNet2DModel(**params_dict)

    def forward(self, sample, timestep, conditioning):
        """
        Forward pass of the Conditional U-Net.

        Args:
            sample (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            timestep (torch.Tensor): Timestep tensor for the diffusion process.
            conditioning (torch.Tensor): Conditioning tensor of the same shape as `sample`.

        Returns:
            torch.Tensor: The output prediction from the U-Net model.

        Raises:
            AssertionError: If `sample` and `conditioning` are not square matrices with matching shapes.
        """
        # assert that both sample and conditioning are square matrices with the same shape
        assert (
            sample.shape[2]
            == sample.shape[3]
            == conditioning.shape[2]
            == conditioning.shape[3]
        ), (
            f"Shape mismatch, sample shape: {sample.shape}, conditioning shape: {conditioning.shape}"
        )

        net_input = torch.cat((sample, conditioning), 1)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input.contiguous(), timestep.contiguous())

    def device(self):
        return self.model.device

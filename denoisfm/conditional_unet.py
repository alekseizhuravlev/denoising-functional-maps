import torch
from torch import nn
from torch.nn import functional as F

from diffusers import UNet2DModel


class ConditionalUnet(nn.Module):
  def __init__(self, params_dict):
    super().__init__()

    self.model = UNet2DModel(**params_dict)


  def forward(self, sample, timestep, conditioning):

    # assert that both sample and conditioning are square matrices with the same shape
    assert sample.shape[2] == sample.shape[3] == conditioning.shape[2] == conditioning.shape[3],\
      f"Shape mismatch, sample shape: {sample.shape}, conditioning shape: {conditioning.shape}"
    
    net_input = torch.cat((sample, conditioning), 1) 

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input.contiguous(), timestep.contiguous())

  def device(self):
    return self.model.device
  
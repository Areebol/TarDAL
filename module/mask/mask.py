import logging
import torch
import torch.nn as nn
from torch import Tensor

class SoftMask(nn.Module):
    r"""
    Use to generate fused mask for ir/iv imag.
    ir + vi -> ir mask + vi mask
    """
    def __init__(self):
        super(SoftMask, self).__init__()
        logging.info("init soft mask for generator")
        self.mask = nn.Sequential(
            nn.Conv2d(2, 2, (3, 3), (1, 1), 1),
            nn.BatchNorm2d(2),
            nn.Sigmoid()
        )

    def forward(self, ir: Tensor, vi: Tensor) -> Tensor:
        src = torch.cat([ir, vi], dim=1)
        mask = self.mask(src)
        return mask
    
    def mask_mean(self, ir: Tensor, vi: Tensor) -> Tensor:
        mask = self.forward(ir,vi)
        ir_mask_mean_value = torch.mean(mask[:,0,:,:])
        vi_mask_mean_value = torch.mean(mask[:,1,:,:])
        return tuple([ir_mask_mean_value, vi_mask_mean_value])

import torch
import torch.nn as nn
from torch.nn import Module
from typing import Tuple, Union
from torch import Tensor

class Warp(Module):
    r"""
    Downsizes the image by removing a defined portion of the center and merging the remaining corners.

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H', W')` where :math:`H' = H - 2 * size` and :math:`W' = W - 2 * size`.

    Args:
        size (int): Size of the portion to be removed from the center of the image.

    Examples::
        >>> import torch
        >>> from custom_modules import Warp
        >>> input = torch.randn(2, 3, 500, 500)
        >>> downsizer = Warp(size=100)
        >>> output = downsizer(input)
        >>> output.shape
        torch.Size([2, 3, 300, 300])
    """

    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def forward(self, x: Tensor) -> Tensor:
        """
        Downsizes the image by removing a defined portion of the center and merging the remaining corners.

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W).

        Returns:
            Tensor: Downsize tensor with the specified portion removed from the center.
        """
        N, C, H, W = x.shape
        
        target_H = H - self.size
        target_W = W - self.size
        # Calculate the starting and ending indices for the remaining corners
        start_H = self.size
        end_H = H - self.size
        start_W = self.size
        end_W = W - self.size
        
        # Slice the image to remove the center portion
        top_left = x[:, :, :target_H//2, :target_W//2]
        top_right = x[:, :, :target_H//2, target_W//2 + self.size:]
        bottom_left = x[:, :, target_H//2 + self.size:, :target_W//2]
        bottom_right = x[:, :, target_H//2 + self.size:, target_W//2 + self.size:]
                
        top = torch.cat((top_left, top_right), dim=3)
        bottom = torch.cat((bottom_left, bottom_right), dim=3)
        
        # Concatenate the remaining corners to create the modified image
        modified_image = torch.cat((top, bottom), dim=2)
        
        return modified_image

    def extra_repr(self) -> str:
        return f'Removed Size: {self.size}'


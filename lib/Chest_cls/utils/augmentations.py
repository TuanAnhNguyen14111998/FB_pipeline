# -*- coding: utf-8 -*-
"""
Created on 7/09/2020 2:23 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import numpy as np

# Third party imports
import torch
import torch.nn as nn
import kornia as K

# Local application imports


class ToTensor(nn.Module):
    """Module to cast numpy image to torch.Tensor."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: np.ndarray) -> torch.Tensor:
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 3 and x.shape[0] == 1, x.shape
        return torch.tensor(x).unsqueeze(0)  # 1xCxHxW


class XRayResizer(nn.Module):
    """Module to image resize spatial resolution."""

    def __init__(self, size: int) -> None:
        super().__init__()
        self._size: int = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 4, x.shape
        return K.resize(x, self._size)


class XrayRandomHorizontalFlip(nn.Module):
    def __init__(self, p: float) -> None:
        super().__init__()
        self._p: float = p
        self.tfs = K.augmentation.RandomHorizontalFlip(p=self._p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        assert len(x.shape) == 4, x.shape
        return self.tfs(x)


class Squeeze(nn.Module):
    """Module to squeeze tensor dimensions."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        if len(x.shape) == 2:
            return torch.unsqueeze(x, dim=0)
        else:
            assert len(x.shape) == 4, x.shape
            return torch.squeeze(x, dim=1)  # CxHxW
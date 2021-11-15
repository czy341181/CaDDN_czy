import torch
import torch.nn as nn
from easydict import EasyDict as edict
import numpy as np


class Conv2DCollapse(nn.Module):

    def __init__(self, model_cfg):
        """
        Initializes 3D convolution collapse module
        Args:
            channels [int]: Number of feature channels
            num_heights [int]: Number of height planes in voxel grid
        """
        super().__init__()

        grid_size = model_cfg['grid_size']
        self.model_cfg = model_cfg['conv2dcollapse']
        self.num_heights = grid_size[-1]
        self.num_bev_features = self.model_cfg['num_bev_features']
        self.block = BasicBlock2D(in_channels=self.num_bev_features * self.num_heights,
                                  out_channels=self.num_bev_features,
                                  **self.model_cfg['ARGS'])

    def forward(self, batch_dict, istrain=True):
        """
        Collapses voxel features to BEV via concatenation and channel reduction
        Args:
            batch_dict:
                voxel_features [torch.Tensor(B, C, Z, Y, X)]: Voxel feature representation
        Returns:
            batch_dict:
                spatial_features [torch.Tensor(B, C, Y, X)]: BEV feature representation
        """
        voxel_features = batch_dict["voxel_features"]
        bev_features = voxel_features.flatten(start_dim=1, end_dim=2)  # (B, C, Z, Y, X) -> (B, C*Z, Y, X)
        bev_features = self.block(bev_features)  # (B, C*Z, Y, X) -> (B, C, Y, X)
        batch_dict["spatial_features"] = bev_features
        return batch_dict


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block for channel reduce
        Args:
            out_channels [int]: Number of output channels of convolutional block
            **kwargs [Dict]: Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features [torch.Tensor(B, C_in, H, W)]: Input features
        Returns:
            x [torch.Tensor(B, C_out, H, W)]: Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x
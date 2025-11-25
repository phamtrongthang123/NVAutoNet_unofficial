"""
BEV Encoder Backbone
Processes fused BEV features with CNN (ResNet-style architecture)
Based on Table 2 in the paper: BEV encoder configuration
"""

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """Basic Conv-BN-ReLU block"""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class BEVResNetBlock(nn.Module):
    """
    Simplified ResNet block for BEV encoder
    Paper uses no residual connections for faster inference (per Table 2 notes)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, num_repeats=1):
        super().__init__()

        layers = []
        for i in range(num_repeats):
            _in_channels = in_channels if i == 0 else out_channels
            _stride = stride if i == 0 else 1
            padding = kernel_size // 2

            layers.append(
                ConvBNReLU(_in_channels, out_channels, kernel_size, _stride, padding)
            )

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BEVResNet(nn.Module):
    """
    BEV Feature Encoder

    Based on Table 2 from paper:
    - Input: 64x360 (for polar) or custom size (for Cartesian)
    - 3 blocks with kernel size 3
    - Strides: 1-2-2
    - Repeats: 4-4-4
    - Channels: 64-128-256

    For nuScenes Cartesian implementation, we adapt the input size
    but keep the same architectural principles.
    """

    def __init__(self,
                 in_channels=80,
                 block_channels=[64, 128, 256],
                 block_strides=[1, 2, 2],
                 block_repeats=[4, 4, 4],
                 kernel_size=3,
                 out_indices=(0, 1, 2)):
        """
        Args:
            in_channels: Input BEV feature channels from view transformer
            block_channels: Output channels for each block
            block_strides: Stride for each block
            block_repeats: Number of conv layers in each block
            kernel_size: Kernel size (3 for all blocks per paper)
            out_indices: Which block outputs to return (for FPN-like multi-scale)
        """
        super().__init__()

        self.in_channels = in_channels
        self.block_channels = block_channels
        self.out_indices = out_indices

        # Build blocks
        self.blocks = nn.ModuleList()
        prev_channels = in_channels

        for i, (channels, stride, repeats) in enumerate(
            zip(block_channels, block_strides, block_repeats)
        ):
            block = BEVResNetBlock(
                prev_channels, channels, kernel_size, stride, repeats
            )
            self.blocks.append(block)
            prev_channels = channels

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W) BEV features

        Returns:
            outs: List of multi-scale features if len(out_indices) > 1,
                  else single tensor
        """
        outs = []

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        return outs


class BEVResNetFPN(nn.Module):
    """
    BEV Encoder with Feature Pyramid Network
    For multi-scale feature extraction
    """

    def __init__(self,
                 in_channels=80,
                 block_channels=[64, 128, 256],
                 fpn_channels=256):
        super().__init__()

        # Backbone
        self.backbone = BEVResNet(
            in_channels=in_channels,
            block_channels=block_channels,
            out_indices=(0, 1, 2)
        )

        # FPN lateral connections
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(channels, fpn_channels, 1)
            for channels in block_channels
        ])

        # FPN output convs
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1)
            for _ in block_channels
        ])

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, H, W)

        Returns:
            fpn_features: List of (B, fpn_channels, H_i, W_i)
        """
        # Get multi-scale features from backbone
        backbone_feats = self.backbone(x)

        # Build FPN
        laterals = [
            lateral_conv(feat)
            for lateral_conv, feat in zip(self.lateral_convs, backbone_feats)
        ]

        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += nn.functional.interpolate(
                laterals[i], scale_factor=2, mode='nearest'
            )

        # Output convs
        fpn_features = [
            fpn_conv(lateral)
            for fpn_conv, lateral in zip(self.fpn_convs, laterals)
        ]

        return fpn_features

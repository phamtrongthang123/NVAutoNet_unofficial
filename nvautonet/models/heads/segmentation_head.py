"""
BEV Segmentation Head
For drivable area and lane segmentation (proxy for freespace in paper)
Uses nuScenes map annotations
"""

import torch
import torch.nn as nn


class BEVSegmentationHead(nn.Module):
    """
    Simple segmentation head for BEV semantic segmentation.

    Predicts:
    - Drivable area
    - Lane markings
    - Pedestrian crossings
    etc. (using nuScenes map data)
    """

    def __init__(self,
                 in_channels=256,
                 num_classes=3,  # drivable, lane, background
                 hidden_channels=128):
        """
        Args:
            in_channels: BEV feature channels
            num_classes: Number of segmentation classes
            hidden_channels: Hidden layer channels
        """
        super().__init__()

        self.num_classes = num_classes

        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_classes, 1)
        )

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: (B, C, H, W) BEV features

        Returns:
            seg_logits: (B, num_classes, H, W)
        """
        return self.decoder(bev_feat)

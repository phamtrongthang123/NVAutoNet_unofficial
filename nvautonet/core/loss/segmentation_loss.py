"""Segmentation Loss for BEV semantic segmentation"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    """
    Simple segmentation loss for BEV maps.
    Uses cross-entropy with optional class weighting.
    """

    def __init__(self, num_classes=3, ignore_index=255, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None

    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, num_classes, H, W) segmentation logits
            targets: (B, H, W) segmentation labels

        Returns:
            loss_dict: Dict containing segmentation loss
        """
        loss = F.cross_entropy(
            predictions,
            targets,
            weight=self.class_weights,
            ignore_index=self.ignore_index
        )

        return {'loss_seg': loss}

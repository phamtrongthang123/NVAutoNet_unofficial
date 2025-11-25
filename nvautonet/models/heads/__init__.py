"""Detection and Perception Heads"""

from .detection_head import NVAutoNetDetectionHead
from .segmentation_head import BEVSegmentationHead

__all__ = [
    'NVAutoNetDetectionHead',
    'BEVSegmentationHead'
]

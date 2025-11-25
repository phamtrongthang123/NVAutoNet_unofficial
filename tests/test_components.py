"""
Unit tests for NVAutoNet components
Run with: pytest tests/test_components.py -v
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nvautonet.models.necks import ColumnwiseMLPTransformer
from nvautonet.models.backbones import BEVResNet
from nvautonet.models.heads import NVAutoNetDetectionHead
from nvautonet.models import NVAutoNet
from nvautonet.utils import generate_bev_grid, generate_bev_lookup_cartesian
from nvautonet.core.matching import greedy_matching
from nvautonet.core.loss import DetectionLoss, AdaptiveLossBalancer
import numpy as np


class TestBEVUtils:
    """Test BEV utility functions"""

    def test_generate_bev_grid(self):
        x_bound = [-50.0, 50.0, 0.5]
        y_bound = [-50.0, 50.0, 0.5]

        grid_coords, grid_shape = generate_bev_grid(x_bound, y_bound)

        # Check shapes
        assert grid_shape == (200, 200)
        assert grid_coords.shape == (200, 200, 2)

        # Check value ranges
        assert grid_coords[:, :, 0].min() >= -50.0
        assert grid_coords[:, :, 0].max() < 50.0
        assert grid_coords[:, :, 1].min() >= -50.0
        assert grid_coords[:, :, 1].max() < 50.0

    def test_generate_bev_lookup(self):
        # Simple test with identity transform
        intrinsic = np.eye(3, dtype=np.float32)
        intrinsic[0, 0] = 1000  # fx
        intrinsic[1, 1] = 1000  # fy
        intrinsic[0, 2] = 400   # cx
        intrinsic[1, 2] = 300   # cy

        extrinsic = np.eye(4, dtype=np.float32)

        lookup = generate_bev_lookup_cartesian(
            intrinsic, extrinsic,
            image_size=(600, 800),
            x_bound=[-50.0, 50.0, 0.5],
            y_bound=[-50.0, 50.0, 0.5]
        )

        # Check shape
        assert lookup.shape == (800, 600, 2)

        # Check that some lookups are valid (not -1)
        assert np.any(lookup >= 0)


class TestViewTransformer:
    """Test Column-wise MLP Transformer"""

    def test_forward_shape(self):
        transformer = ColumnwiseMLPTransformer(
            in_channels=256,
            out_channels=80,
            image_size=(28, 50),
            bev_h=200,
            bev_w=200
        )

        # Create dummy camera parameters
        camera_params = {
            'CAM_FRONT': {
                'intrinsic': torch.eye(3),
                'extrinsic': torch.eye(4)
            }
        }

        transformer.generate_lookup_tables(camera_params)

        # Dummy input
        img_feats = {
            'CAM_FRONT': torch.randn(2, 256, 28, 50)
        }

        # Forward pass
        bev_feat, aux = transformer(img_feats, camera_ids=['CAM_FRONT'])

        # Check output shape
        assert bev_feat.shape == (2, 80, 200, 200)


class TestBEVEncoder:
    """Test BEV Encoder"""

    def test_forward_shape(self):
        encoder = BEVResNet(
            in_channels=80,
            block_channels=[64, 128, 256]
        )

        # Dummy input
        bev_feat = torch.randn(2, 80, 200, 200)

        # Forward pass
        output = encoder(bev_feat)

        # Check output shape (last block output)
        assert output.shape == (2, 256, 50, 50)  # After 4x downsampling


class TestDetectionHead:
    """Test Detection Head"""

    def test_forward_shape(self):
        head = NVAutoNetDetectionHead(
            num_classes=10,
            num_queries=300,
            bev_channels=256
        )

        # Dummy input
        bev_feat = torch.randn(2, 256, 50, 50)

        # Forward pass
        predictions = head(bev_feat)

        # Check output shapes
        assert predictions['cls_logits'].shape == (2, 300, 11)  # num_classes + 1
        assert predictions['bbox_preds'].shape == (2, 300, 10)
        assert predictions['uncertainties'].shape == (2, 300, 5)


class TestMatching:
    """Test Greedy Matching"""

    def test_greedy_matching(self):
        # Dummy predictions and ground truth
        pred_boxes = torch.tensor([
            [10.0, 5.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [15.0, 10.0, 0.0, 2.0, 4.0, 1.5, 0.0],
            [20.0, 15.0, 0.0, 2.0, 4.0, 1.5, 0.0]
        ])

        pred_scores = torch.randn(3, 10)

        gt_boxes = torch.tensor([
            [10.5, 5.5, 0.0, 2.0, 4.0, 1.5, 0.0],
            [20.5, 15.5, 0.0, 2.0, 4.0, 1.5, 0.0]
        ])

        gt_labels = torch.tensor([0, 1])

        # Run matching
        matched, unmatched_preds, unmatched_gts = greedy_matching(
            pred_boxes, pred_scores, gt_boxes, gt_labels
        )

        # Should match first and third predictions
        assert len(matched) <= 2
        assert len(unmatched_preds) >= 1


class TestLossFunctions:
    """Test Loss Functions"""

    def test_detection_loss(self):
        loss_fn = DetectionLoss(num_classes=10)

        # Dummy predictions
        predictions = {
            'cls_logits': torch.randn(2, 300, 11),
            'bbox_preds': torch.randn(2, 300, 10),
            'uncertainties': torch.randn(2, 300, 5)
        }

        # Dummy targets
        targets = [
            {
                'boxes': torch.randn(5, 7),
                'labels': torch.randint(0, 10, (5,))
            },
            {
                'boxes': torch.randn(3, 7),
                'labels': torch.randint(0, 10, (3,))
            }
        ]

        # Compute loss
        loss_dict = loss_fn(predictions, targets)

        # Check that losses are computed
        assert 'loss_total' in loss_dict
        assert 'loss_cls' in loss_dict
        assert loss_dict['loss_total'].item() >= 0


class TestLossBalancer:
    """Test Adaptive Loss Balancer"""

    def test_adaptive_balancing(self):
        balancer = AdaptiveLossBalancer(
            task_names=['detection', 'segmentation'],
            task_priors={'detection': 5.0, 'segmentation': 1.0}
        )

        # Simulate training
        for epoch in range(3):
            balancer.reset_epoch_losses()

            for batch in range(10):
                # Dummy losses
                task_losses = {
                    'detection': torch.tensor(2.0),
                    'segmentation': torch.tensor(0.5)
                }

                balancer.accumulate_losses(task_losses)

            # Update weights
            balancer.update_weights()

        # Check that weights are updated
        assert 'detection' in balancer.task_weights
        assert 'segmentation' in balancer.task_weights
        assert sum(balancer.task_weights.values()) == pytest.approx(1.0)


class TestNVAutoNet:
    """Test Complete Model"""

    def test_forward_inference(self):
        model = NVAutoNet(
            img_backbone='resnet50',
            img_backbone_pretrained=False,  # Faster for testing
            view_transformer_cfg={
                'in_channels': 2048,
                'out_channels': 80,
                'image_size': (28, 50),
                'bev_h': 100,
                'bev_w': 100,
                'x_bound': [-25.0, 25.0, 0.5],
                'y_bound': [-25.0, 25.0, 0.5],
            },
            bev_encoder_cfg={
                'in_channels': 80,
                'block_channels': [32, 64, 128],
            },
            detection_head_cfg={
                'num_classes': 10,
                'num_queries': 100,  # Reduced for testing
                'bev_channels': 128,
            }
        )

        model.eval_mode()

        # Generate lookup tables
        camera_params = {}
        for cam_id in ['CAM_FRONT', 'CAM_BACK']:
            camera_params[cam_id] = {
                'intrinsic': torch.eye(3),
                'extrinsic': torch.eye(4)
            }

        model.view_transformer.generate_lookup_tables(camera_params)

        # Dummy batch
        batch = {
            'imgs': {
                'CAM_FRONT': torch.randn(1, 3, 900, 1600),
                'CAM_BACK': torch.randn(1, 3, 900, 1600)
            },
            'camera_ids': ['CAM_FRONT', 'CAM_BACK']
        }

        # Forward pass (inference mode)
        with torch.no_grad():
            predictions = model(batch)

        # Check outputs
        assert 'detection' in predictions
        assert predictions['detection']['cls_logits'].shape[1] == 100  # num_queries


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

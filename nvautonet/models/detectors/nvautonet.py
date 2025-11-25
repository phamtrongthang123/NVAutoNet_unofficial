"""
NVAutoNet: Complete End-to-End Detector
Combines all components: 2D backbone → View Transformer → BEV Encoder → Heads
"""

import torch
import torch.nn as nn
import torchvision

from ..necks import ColumnwiseMLPTransformer
from ..backbones import BEVResNet
from ..heads import NVAutoNetDetectionHead, BEVSegmentationHead
from ...core.loss import DetectionLoss, SegmentationLoss, AdaptiveLossBalancer


class NVAutoNet(nn.Module):
    """
    Complete NVAutoNet model for 3D perception.

    Pipeline:
        Multi-camera Images [B, N_cams, 3, H, W]
            ↓
        2D CNN Backbones → [B, N_cams, C, h, w]
            ↓
        Column-wise MLP Transformer → [B, C_bev, H_bev, W_bev]
            ↓
        BEV Encoder → [B, C_enc, H_enc, W_enc]
            ↓
        Task Heads → {detection, segmentation, ...}
    """

    def __init__(self,
                 # 2D Backbone config
                 img_backbone='resnet50',
                 img_backbone_pretrained=True,
                 img_feat_channels=2048,

                 # View Transformer config
                 view_transformer_cfg=None,

                 # BEV Encoder config
                 bev_encoder_cfg=None,

                 # Detection Head config
                 detection_head_cfg=None,

                 # Segmentation Head config (optional)
                 segmentation_head_cfg=None,

                 # Loss config
                 loss_cfg=None,

                 # Multi-task learning
                 use_adaptive_loss_balancing=True,
                 task_priors=None):
        """
        Args:
            img_backbone: 2D CNN backbone name ('resnet50', 'resnet101', etc.)
            img_backbone_pretrained: Whether to use ImageNet pretrained weights
            img_feat_channels: Output channels from 2D backbone
            view_transformer_cfg: Config dict for ColumnwiseMLPTransformer
            bev_encoder_cfg: Config dict for BEV encoder
            detection_head_cfg: Config dict for detection head
            segmentation_head_cfg: Config dict for segmentation head (None to disable)
            loss_cfg: Config dict for losses
            use_adaptive_loss_balancing: Whether to use adaptive loss balancing
            task_priors: Task weight priors for loss balancing
        """
        super().__init__()

        # === 2D Image Backbone ===
        self.img_backbone = self._build_img_backbone(
            img_backbone, img_backbone_pretrained
        )

        # === View Transformer ===
        if view_transformer_cfg is None:
            view_transformer_cfg = {
                'in_channels': img_feat_channels,
                'out_channels': 80,
                'image_size': (28, 50),  # After backbone downsampling (900/32, 1600/32)
                'bev_h': 200,
                'bev_w': 200,
                'x_bound': [-50.0, 50.0, 0.5],
                'y_bound': [-50.0, 50.0, 0.5],
            }

        self.view_transformer = ColumnwiseMLPTransformer(**view_transformer_cfg)

        # === BEV Encoder ===
        if bev_encoder_cfg is None:
            bev_encoder_cfg = {
                'in_channels': view_transformer_cfg['out_channels'],
                'block_channels': [64, 128, 256],
                'out_indices': (2,),  # Only return last block output
            }

        self.bev_encoder = BEVResNet(**bev_encoder_cfg)

        # === Task Heads ===
        if detection_head_cfg is None:
            detection_head_cfg = {
                'num_classes': 10,
                'bev_channels': bev_encoder_cfg['block_channels'][-1],
            }

        self.detection_head = NVAutoNetDetectionHead(**detection_head_cfg)

        self.use_segmentation = segmentation_head_cfg is not None
        if self.use_segmentation:
            self.segmentation_head = BEVSegmentationHead(**segmentation_head_cfg)

        # === Loss Functions ===
        if loss_cfg is None:
            loss_cfg = {}

        self.detection_loss = DetectionLoss(**loss_cfg.get('detection', {}))

        if self.use_segmentation:
            self.segmentation_loss = SegmentationLoss(**loss_cfg.get('segmentation', {}))

        # === Loss Balancer ===
        task_names = ['detection']
        if self.use_segmentation:
            task_names.append('segmentation')

        if use_adaptive_loss_balancing:
            self.loss_balancer = AdaptiveLossBalancer(
                task_names=task_names,
                task_priors=task_priors
            )
        else:
            from ...core.loss import FixedLossBalancer
            self.loss_balancer = FixedLossBalancer(task_names=task_names)

        # Training mode flag
        self.training_mode = True

    def _build_img_backbone(self, backbone_name, pretrained):
        """Build 2D image backbone (ResNet)"""

        if 'resnet' in backbone_name:
            if backbone_name == 'resnet18':
                backbone = torchvision.models.resnet18(pretrained=pretrained)
                out_channels = 512
            elif backbone_name == 'resnet34':
                backbone = torchvision.models.resnet34(pretrained=pretrained)
                out_channels = 512
            elif backbone_name == 'resnet50':
                backbone = torchvision.models.resnet50(pretrained=pretrained)
                out_channels = 2048
            elif backbone_name == 'resnet101':
                backbone = torchvision.models.resnet101(pretrained=pretrained)
                out_channels = 2048
            else:
                raise ValueError(f"Unsupported ResNet variant: {backbone_name}")

            # Remove the final FC and avgpool layers
            # Keep only conv layers
            modules = list(backbone.children())[:-2]
            backbone = nn.Sequential(*modules)

            return backbone
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def extract_img_features(self, imgs):
        """
        Extract features from multi-camera images.

        Args:
            imgs: (B, N_cams, 3, H, W) or Dict[cam_id -> (B, 3, H, W)]

        Returns:
            img_feats: (B, N_cams, C, h, w) or Dict[cam_id -> (B, C, h, w)]
        """
        if isinstance(imgs, dict):
            # Dict format
            img_feats = {}
            for cam_id, img in imgs.items():
                B, C, H, W = img.shape
                feat = self.img_backbone(img)  # (B, C_feat, h, w)
                img_feats[cam_id] = feat
            return img_feats

        else:
            # Tensor format: (B, N_cams, 3, H, W)
            B, N, C, H, W = imgs.shape

            # Reshape to (B*N, 3, H, W) for batch processing
            imgs_flat = imgs.view(B * N, C, H, W)

            # Extract features
            feats_flat = self.img_backbone(imgs_flat)  # (B*N, C_feat, h, w)

            # Reshape back to (B, N, C_feat, h, w)
            C_feat, h, w = feats_flat.shape[1:]
            img_feats = feats_flat.view(B, N, C_feat, h, w)

            return img_feats

    def forward(self, batch):
        """
        Forward pass.

        Args:
            batch: Dict containing:
                - imgs: (B, N_cams, 3, H, W) or Dict[cam_id -> (B, 3, H, W)]
                - camera_ids: List of camera IDs (if imgs is tensor)
                - targets: List of target dicts (for training)
                    - boxes: (N_gt, 7+)
                    - labels: (N_gt,)
                    - seg_maps: (H_bev, W_bev) if using segmentation

        Returns:
            If training: loss_dict
            If inference: predictions dict
        """
        imgs = batch['imgs']
        camera_ids = batch.get('camera_ids', None)

        # Extract 2D features
        img_feats = self.extract_img_features(imgs)

        # Transform to BEV
        bev_feat, aux_outputs = self.view_transformer(img_feats, camera_ids)

        # Encode BEV features
        bev_feat_encoded = self.bev_encoder(bev_feat)

        # Task-specific heads
        det_preds = self.detection_head(bev_feat_encoded)

        predictions = {
            'detection': det_preds,
            'bev_feat': bev_feat,
            'aux_outputs': aux_outputs
        }

        if self.use_segmentation:
            seg_preds = self.segmentation_head(bev_feat_encoded)
            predictions['segmentation'] = seg_preds

        # Compute losses if training
        if self.training_mode and 'targets' in batch:
            return self.compute_losses(predictions, batch['targets'])
        else:
            return predictions

    def compute_losses(self, predictions, targets):
        """
        Compute multi-task losses.

        Args:
            predictions: Dict of task predictions
            targets: List of target dicts

        Returns:
            loss_dict: Dict of losses
        """
        task_losses = {}

        # Detection loss
        det_preds = predictions['detection']
        det_loss_dict = self.detection_loss(det_preds, targets)
        task_losses['detection'] = det_loss_dict['loss_total']

        # Segmentation loss (if enabled)
        if self.use_segmentation and 'seg_maps' in targets[0]:
            seg_preds = predictions['segmentation']
            seg_targets = torch.stack([t['seg_maps'] for t in targets])
            seg_loss_dict = self.segmentation_loss(seg_preds, seg_targets)
            task_losses['segmentation'] = seg_loss_dict['loss_seg']

        # Apply loss balancing
        weighted_loss, loss_dict = self.loss_balancer.get_weighted_loss(task_losses)

        # Add individual task losses for logging
        loss_dict.update(det_loss_dict)
        if self.use_segmentation:
            loss_dict.update(seg_loss_dict)

        # Accumulate losses for adaptive balancing
        self.loss_balancer.accumulate_losses(task_losses)

        return loss_dict

    def train_mode(self):
        """Set to training mode"""
        self.train()
        self.training_mode = True

    def eval_mode(self):
        """Set to evaluation mode"""
        self.eval()
        self.training_mode = False

"""
3D Object Detection Head
Set prediction approach without NMS (DETR-style)
Based on Section 4.1 of the paper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class NVAutoNetDetectionHead(nn.Module):
    """
    3D Object Detection Head using set prediction.

    Key features:
    - Fixed number of object queries (300 per paper)
    - No NMS post-processing needed
    - Predicts 9-DOF bounding boxes: position (r, a, e), size (dx, dy, dz),
      rotation (sin/cos of yaw, pitch, roll)
    - Uncertainty prediction for adaptive weighting

    Architecture:
        BEV Features [B, C, H, W]
            ↓
        Object Queries [num_queries, embed_dim] (learnable)
            ↓
        Cross-attention with BEV features
            ↓
        Classification Head → [B, num_queries, num_classes+1]
        Regression Head → [B, num_queries, 9]
        Uncertainty Head → [B, num_queries, 5]
    """

    def __init__(self,
                 num_classes=10,
                 num_queries=300,
                 embed_dims=256,
                 bev_channels=256,
                 num_reg_fcs=2,
                 use_uncertainty=True):
        """
        Args:
            num_classes: Number of object classes (10 for nuScenes)
            num_queries: Number of object queries (300 per paper)
            embed_dims: Embedding dimension for queries
            bev_channels: BEV feature channels
            num_reg_fcs: Number of FC layers for regression head
            use_uncertainty: Whether to predict uncertainty for adaptive loss weighting
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_queries = num_queries
        self.embed_dims = embed_dims
        self.use_uncertainty = use_uncertainty

        # Learnable object queries
        self.query_embedding = nn.Embedding(num_queries, embed_dims)

        # Project BEV features to query dimension
        self.bev_proj = nn.Conv2d(bev_channels, embed_dims, 1)

        # Cross-attention layers
        self.cross_attention = nn.MultiheadAttention(
            embed_dims, num_heads=8, dropout=0.1, batch_first=True
        )

        # Self-attention for query refinement
        self.self_attention = nn.MultiheadAttention(
            embed_dims, num_heads=8, dropout=0.1, batch_first=True
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(embed_dims * 4, embed_dims),
            nn.Dropout(0.1)
        )

        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        # Classification head (num_classes + 1 for background/no object)
        self.cls_head = nn.Linear(embed_dims, num_classes + 1)

        # Regression head for 9-DOF bounding boxes
        # [r, a, e, dx, dy, dz, sin(yaw), cos(yaw), sin(pitch), cos(pitch), sin(roll), cos(roll)]
        # Simplified to: [x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]
        reg_layers = []
        for i in range(num_reg_fcs):
            reg_layers.extend([
                nn.Linear(embed_dims if i == 0 else 256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        reg_layers.append(nn.Linear(256, 10))  # x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy
        self.reg_head = nn.Sequential(*reg_layers)

        # Uncertainty head (5 values: σ_r, σ_a, σ_e, σ_s, σ_o)
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(embed_dims, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 5),
                nn.Softplus()  # Ensure positive uncertainty values
            )

        self._init_weights()

    def _init_weights(self):
        # Initialize query embeddings
        nn.init.normal_(self.query_embedding.weight, std=0.01)

        # Initialize classification head with bias for stable training
        nn.init.constant_(self.cls_head.bias, -math.log((1 - 0.01) / 0.01))

        # Initialize regression head
        for m in self.reg_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: (B, C, H, W) BEV features from encoder

        Returns:
            predictions: Dict containing:
                - cls_logits: (B, num_queries, num_classes+1)
                - bbox_preds: (B, num_queries, 10)
                - uncertainties: (B, num_queries, 5) if use_uncertainty
        """
        B, C, H, W = bev_feat.shape

        # Project BEV features
        bev_feat = self.bev_proj(bev_feat)  # (B, embed_dims, H, W)

        # Flatten BEV features for attention
        # (B, embed_dims, H, W) -> (B, H*W, embed_dims)
        bev_feat_flat = bev_feat.flatten(2).permute(0, 2, 1)

        # Get object queries: (num_queries, embed_dims) -> (B, num_queries, embed_dims)
        queries = self.query_embedding.weight.unsqueeze(0).repeat(B, 1, 1)

        # Cross-attention: queries attend to BEV features
        queries_attn, _ = self.cross_attention(
            query=queries,
            key=bev_feat_flat,
            value=bev_feat_flat
        )
        queries = self.norm1(queries + queries_attn)

        # Self-attention: queries attend to each other
        queries_self, _ = self.self_attention(
            query=queries,
            key=queries,
            value=queries
        )
        queries = self.norm2(queries + queries_self)

        # Feed-forward network
        queries_ffn = self.ffn(queries)
        queries = self.norm3(queries + queries_ffn)

        # Prediction heads
        cls_logits = self.cls_head(queries)  # (B, num_queries, num_classes+1)
        bbox_preds = self.reg_head(queries)  # (B, num_queries, 10)

        predictions = {
            'cls_logits': cls_logits,
            'bbox_preds': bbox_preds
        }

        if self.use_uncertainty:
            uncertainties = self.uncertainty_head(queries)  # (B, num_queries, 5)
            predictions['uncertainties'] = uncertainties

        return predictions

    def decode_bbox(self, bbox_preds, bev_shape):
        """
        Decode bbox predictions to absolute coordinates.

        Args:
            bbox_preds: (B, num_queries, 10) raw predictions
            bev_shape: (H, W, x_bound, y_bound) BEV configuration

        Returns:
            decoded_bboxes: (B, num_queries, 10) decoded boxes
        """
        # Extract predictions
        # bbox_preds: [x, y, z, w, l, h, sin(yaw), cos(yaw), vx, vy]

        # Apply sigmoid to normalize center coordinates
        xy = torch.sigmoid(bbox_preds[..., :2])  # (B, num_queries, 2)

        # Scale to BEV dimensions
        H, W, x_bound, y_bound = bev_shape
        x_min, x_max, _ = x_bound
        y_min, y_max, _ = y_bound

        xy[..., 0] = xy[..., 0] * (x_max - x_min) + x_min
        xy[..., 1] = xy[..., 1] * (y_max - y_min) + y_min

        # Height (z) can be directly regressed
        z = bbox_preds[..., 2:3]

        # Dimensions (w, l, h) - apply exp for positive values
        dims = torch.exp(bbox_preds[..., 3:6])

        # Normalize rotation (sin, cos)
        sin_yaw = bbox_preds[..., 6:7]
        cos_yaw = bbox_preds[..., 7:8]
        norm = torch.sqrt(sin_yaw**2 + cos_yaw**2 + 1e-8)
        sin_yaw = sin_yaw / norm
        cos_yaw = cos_yaw / norm

        # Velocity
        velocity = bbox_preds[..., 8:10]

        decoded_bboxes = torch.cat([
            xy, z, dims, sin_yaw, cos_yaw, velocity
        ], dim=-1)

        return decoded_bboxes


class MultiTaskHead(nn.Module):
    """
    Wrapper for multiple task heads (detection + segmentation + etc.)
    """

    def __init__(self,
                 detection_head_cfg,
                 segmentation_head_cfg=None):
        super().__init__()

        self.detection_head = NVAutoNetDetectionHead(**detection_head_cfg)

        self.has_segmentation = segmentation_head_cfg is not None
        if self.has_segmentation:
            from .segmentation_head import BEVSegmentationHead
            self.segmentation_head = BEVSegmentationHead(**segmentation_head_cfg)

    def forward(self, bev_feat):
        """
        Args:
            bev_feat: (B, C, H, W) BEV features

        Returns:
            outputs: Dict with task-specific predictions
        """
        outputs = {}

        # Detection
        det_preds = self.detection_head(bev_feat)
        outputs['detection'] = det_preds

        # Segmentation (optional)
        if self.has_segmentation:
            seg_preds = self.segmentation_head(bev_feat)
            outputs['segmentation'] = seg_preds

        return outputs

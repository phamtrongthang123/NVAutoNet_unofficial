"""
Column-wise MLP View Transformer
Core innovation of NVAutoNet for 2D-to-BEV transformation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nvautonet.utils import generate_bev_lookup_cartesian_torch


class ColumnwiseMLPTransformer(nn.Module):
    """
    Column-wise MLP for 2D image features to BEV transformation.

    Key Innovation:
    - Processes each image column independently with shared MLP
    - Incorporates camera intrinsic/extrinsic parameters via BEV lookup tables
    - More efficient than full image vectorization
    - Enables generalization across different camera setups

    Architecture:
        Image Features [B, C, H, W]
            ↓ (per column)
        Column Features [B, C*H, W] → treat each column as a vector
            ↓ (shared MLP)
        Pseudo-BEV Features [B, C_bev*D, W] → D depth bins per column
            ↓ (BEV scatter using lookup table)
        BEV Features [B, C_bev, BEV_H, BEV_W]
    """

    def __init__(self,
                 in_channels=512,
                 out_channels=80,
                 image_size=(900, 1600),
                 bev_h=200,
                 bev_w=200,
                 x_bound=[-50.0, 50.0, 0.5],
                 y_bound=[-50.0, 50.0, 0.5],
                 mlp_hidden_dims=[256, 128],
                 num_depth_bins=64,
                 use_depth_supervision=False):
        """
        Args:
            in_channels: Input feature channels from 2D backbone
            out_channels: Output BEV feature channels
            image_size: (H, W) of input feature maps
            bev_h: BEV grid height
            bev_w: BEV grid width
            x_bound: [x_min, x_max, x_resolution] for BEV grid
            y_bound: [y_min, y_max, y_resolution] for BEV grid
            mlp_hidden_dims: List of hidden dimensions for MLP
            num_depth_bins: Number of depth bins to discretize
            use_depth_supervision: Whether to predict explicit depth
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.image_h, self.image_w = image_size
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_bound = x_bound
        self.y_bound = y_bound
        self.num_depth_bins = num_depth_bins
        self.use_depth_supervision = use_depth_supervision

        # Column-wise MLP
        # Input: Features from entire column [C * H]
        # Output: BEV features for multiple depth bins [out_channels * num_depth_bins]
        mlp_layers = []
        prev_dim = in_channels * self.image_h

        for hidden_dim in mlp_hidden_dims:
            mlp_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim

        # Output layer
        mlp_layers.append(
            nn.Linear(prev_dim, out_channels * num_depth_bins)
        )

        self.column_mlp = nn.Sequential(*mlp_layers)

        # Optional depth prediction head
        if use_depth_supervision:
            self.depth_head = nn.Sequential(
                nn.Linear(prev_dim, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, num_depth_bins)
            )

        # BEV lookup tables will be registered as buffers when provided
        # These are precomputed from camera parameters
        self.bev_lookup_tables = {}

    def register_camera_lookup(self, cam_id, lookup_table):
        """
        Register precomputed BEV lookup table for a camera.

        Args:
            cam_id: Camera identifier (e.g., 'CAM_FRONT')
            lookup_table: (W, H, 2) tensor of BEV indices
        """
        self.register_buffer(
            f'bev_lookup_{cam_id}',
            lookup_table,
            persistent=False
        )
        self.bev_lookup_tables[cam_id] = lookup_table

    def generate_lookup_tables(self, camera_params):
        """
        Generate BEV lookup tables from camera parameters.

        Args:
            camera_params: Dict mapping camera_id -> {
                'intrinsic': (3, 3) tensor,
                'extrinsic': (4, 4) tensor
            }
        """
        for cam_id, params in camera_params.items():
            lookup = generate_bev_lookup_cartesian_torch(
                camera_intrinsic=params['intrinsic'],
                camera_extrinsic=params['extrinsic'],
                image_size=(self.image_h, self.image_w),
                x_bound=self.x_bound,
                y_bound=self.y_bound,
                device=params['intrinsic'].device
            )
            self.register_camera_lookup(cam_id, lookup)

    def forward_single_camera(self, img_feat, lookup_table):
        """
        Process features from a single camera.

        Args:
            img_feat: (B, C, H, W) image features from 2D backbone
            lookup_table: (W, H, 2) BEV indices for each pixel

        Returns:
            bev_feat: (B, out_channels, bev_h, bev_w) BEV features
            depth_logits: (B, W, num_depth_bins) if use_depth_supervision
        """
        B, C, H, W = img_feat.shape

        assert H == self.image_h and W == self.image_w, \
            f"Image size mismatch: expected {(self.image_h, self.image_w)}, got {(H, W)}"

        # Initialize BEV feature map
        bev_feat = torch.zeros(
            B, self.out_channels, self.bev_h, self.bev_w,
            device=img_feat.device,
            dtype=img_feat.dtype
        )

        # Optional depth predictions
        depth_logits = None
        if self.use_depth_supervision:
            depth_logits = []

        # Process each column independently
        # Reshape to group by column: (B, C, H, W) -> (B*W, C*H)
        img_feat_col = img_feat.permute(0, 3, 1, 2).contiguous()  # (B, W, C, H)
        img_feat_col = img_feat_col.view(B * W, C * H)  # (B*W, C*H)

        # Apply MLP: (B*W, C*H) -> (B*W, out_channels * num_depth_bins)
        col_bev_feat = self.column_mlp(img_feat_col)

        # Reshape: (B*W, out_channels * num_depth_bins) -> (B, W, out_channels, num_depth_bins)
        col_bev_feat = col_bev_feat.view(B, W, self.out_channels, self.num_depth_bins)

        # Optional depth prediction
        if self.use_depth_supervision:
            depth_logits = self.depth_head(img_feat_col)  # (B*W, num_depth_bins)
            depth_logits = depth_logits.view(B, W, self.num_depth_bins)

        # Scatter to BEV grid using lookup table
        # For each column and depth bin, lookup tells us where to place features
        for col_idx in range(W):
            for depth_idx in range(self.num_depth_bins):
                # Get BEV positions for this column at this depth
                # lookup_table[col_idx]: (H, 2) - one BEV position per pixel row
                # For simplicity, we use middle row for each column
                # In practice, you might want to aggregate over multiple rows

                mid_row = H // 2
                bev_x_idx = lookup_table[col_idx, mid_row, 0].item()
                bev_y_idx = lookup_table[col_idx, mid_row, 1].item()

                # Skip if outside BEV grid
                if bev_x_idx < 0 or bev_y_idx < 0:
                    continue

                if bev_x_idx >= self.bev_w or bev_y_idx >= self.bev_h:
                    continue

                # Scatter features to BEV grid
                # Accumulate (could also use max or learned fusion)
                bev_feat[:, :, bev_y_idx, bev_x_idx] += col_bev_feat[:, col_idx, :, depth_idx]

        if self.use_depth_supervision:
            return bev_feat, depth_logits
        else:
            return bev_feat, None

    def forward(self, img_feats, camera_ids=None):
        """
        Forward pass for multi-camera BEV transformation.

        Args:
            img_feats: List of (B, N_cams, C, H, W) or
                      Dict mapping camera_id -> (B, C, H, W)
            camera_ids: List of camera identifiers (if img_feats is a list)

        Returns:
            bev_feat: (B, out_channels, bev_h, bev_w) fused BEV features
            aux_outputs: Dict with auxiliary outputs (e.g., depth predictions)
        """
        # Handle different input formats
        if isinstance(img_feats, dict):
            # Dict format: {cam_id: (B, C, H, W)}
            cam_feats_dict = img_feats
        elif isinstance(img_feats, (list, tuple)):
            # List/Tuple format: [(B, C, H, W), ...]
            assert camera_ids is not None, "camera_ids required for list input"
            cam_feats_dict = {
                cam_id: feat for cam_id, feat in zip(camera_ids, img_feats)
            }
        elif isinstance(img_feats, torch.Tensor):
            # Tensor format: (B, N_cams, C, H, W)
            assert camera_ids is not None, "camera_ids required for tensor input"
            B, N, C, H, W = img_feats.shape
            cam_feats_dict = {
                camera_ids[i]: img_feats[:, i] for i in range(N)
            }
        else:
            raise TypeError(f"Unsupported img_feats type: {type(img_feats)}")

        # Get batch size
        B = next(iter(cam_feats_dict.values())).shape[0]

        # Initialize fused BEV feature map
        bev_feat_fused = torch.zeros(
            B, self.out_channels, self.bev_h, self.bev_w,
            device=next(iter(cam_feats_dict.values())).device,
            dtype=next(iter(cam_feats_dict.values())).dtype
        )

        # Auxiliary outputs
        aux_outputs = {'depth_logits': {}}

        # Process each camera
        for cam_id, img_feat in cam_feats_dict.items():
            # Get lookup table for this camera
            if cam_id not in self.bev_lookup_tables:
                raise ValueError(
                    f"BEV lookup table not found for camera {cam_id}. "
                    f"Call generate_lookup_tables() or register_camera_lookup() first."
                )

            lookup_table = self.bev_lookup_tables[cam_id]

            # Transform to BEV
            bev_feat_cam, depth_logits = self.forward_single_camera(img_feat, lookup_table)

            # Fuse (simple addition, could use learned weights)
            bev_feat_fused += bev_feat_cam

            # Store depth predictions
            if depth_logits is not None:
                aux_outputs['depth_logits'][cam_id] = depth_logits

        # Normalize by number of cameras (optional)
        # bev_feat_fused = bev_feat_fused / len(cam_feats_dict)

        return bev_feat_fused, aux_outputs


class LearnedCameraFusion(nn.Module):
    """
    Optional: Learned fusion weights for multi-camera BEV features.
    Instead of simple addition, learn importance weights per camera.
    """

    def __init__(self, num_cameras, bev_channels):
        super().__init__()
        self.num_cameras = num_cameras
        self.bev_channels = bev_channels

        # Learnable fusion weights per camera
        self.fusion_weights = nn.Parameter(torch.ones(num_cameras))

        # Optional: Spatial attention for fusion
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(bev_channels * num_cameras, bev_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(bev_channels, num_cameras, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, bev_feats_list):
        """
        Args:
            bev_feats_list: List of (B, C, H, W) BEV features from each camera

        Returns:
            fused_bev: (B, C, H, W) fused BEV features
        """
        # Stack: (B, N_cams, C, H, W)
        bev_stack = torch.stack(bev_feats_list, dim=1)
        B, N, C, H, W = bev_stack.shape

        # Simple weighted sum
        weights = F.softmax(self.fusion_weights, dim=0)
        weights = weights.view(1, N, 1, 1, 1)

        fused_bev = (bev_stack * weights).sum(dim=1)

        # Optional: Apply spatial attention
        # Concatenate all features: (B, N*C, H, W)
        # bev_concat = bev_stack.view(B, N * C, H, W)
        # attention_weights = self.spatial_attention(bev_concat)  # (B, N, H, W)
        # attention_weights = attention_weights.unsqueeze(2)  # (B, N, 1, H, W)
        # fused_bev = (bev_stack * attention_weights).sum(dim=1)

        return fused_bev

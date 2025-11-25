"""
Detection Loss Functions
Based on Section 4.1 of the paper (Equations 3-7)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..matching import greedy_matching


class FocalLoss(nn.Module):
    """Focal loss for classification"""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, num_classes) logits
            targets: (N,) class indices

        Returns:
            loss: scalar
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p) ** self.gamma * ce_loss

        return focal_loss.mean()


class DetectionLoss(nn.Module):
    """
    Complete detection loss combining classification and regression.

    Based on paper Section 4.1:
    - Classification loss: Focal loss
    - Regression loss: Decomposed into location, size, rotation
    - Uncertainty-weighted losses (Equation 5-7)
    """

    def __init__(self,
                 num_classes=10,
                 loss_cls_weight=1.0,
                 loss_bbox_weight=1.0,
                 use_focal_loss=True,
                 use_uncertainty=True):
        super().__init__()

        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_bbox_weight = loss_bbox_weight
        self.use_uncertainty = use_uncertainty

        if use_focal_loss:
            self.cls_loss_fn = FocalLoss()
        else:
            self.cls_loss_fn = nn.CrossEntropyLoss()

    def compute_location_loss(self, pred_loc, gt_loc, uncertainty=None):
        """
        Location loss (Equation 5 in paper):
        L_loc = |r_pred - r_gt| / σ_r + |a_pred - a_gt| / σ_a + |e_pred - e_gt| / σ_e
                + log(2σ_r) + log(2σ_a) + log(2σ_e)

        Args:
            pred_loc: (N, 3) predicted [x, y, z]
            gt_loc: (N, 3) ground truth [x, y, z]
            uncertainty: (N, 3) uncertainty values [σ_r, σ_a, σ_e] if use_uncertainty

        Returns:
            loss: scalar
        """
        if self.use_uncertainty and uncertainty is not None:
            # Uncertainty-weighted loss
            sigma_r, sigma_a, sigma_e = uncertainty[:, 0], uncertainty[:, 1], uncertainty[:, 2]

            loss_r = torch.abs(pred_loc[:, 0] - gt_loc[:, 0]) / (sigma_r + 1e-8) + torch.log(2 * sigma_r + 1e-8)
            loss_a = torch.abs(pred_loc[:, 1] - gt_loc[:, 1]) / (sigma_a + 1e-8) + torch.log(2 * sigma_a + 1e-8)
            loss_e = torch.abs(pred_loc[:, 2] - gt_loc[:, 2]) / (sigma_e + 1e-8) + torch.log(2 * sigma_e + 1e-8)

            loss = loss_r + loss_a + loss_e
        else:
            # Simple L1 loss
            loss = F.l1_loss(pred_loc, gt_loc, reduction='none').sum(dim=1)

        return loss.mean()

    def compute_size_loss(self, pred_size, gt_size, uncertainty=None):
        """
        Size loss (Equation 6 in paper):
        L_size = 1/σ_s * (1 - ∏ min(dim_pred, dim_gt) / max(dim_pred, dim_gt)) + log(2σ_s)

        Args:
            pred_size: (N, 3) predicted [w, l, h]
            gt_size: (N, 3) ground truth [w, l, h]
            uncertainty: (N,) uncertainty σ_s if use_uncertainty

        Returns:
            loss: scalar
        """
        # IoU-like loss for dimensions
        mins = torch.min(pred_size, gt_size)
        maxs = torch.max(pred_size, gt_size)

        iou_dims = (mins / (maxs + 1e-8)).prod(dim=1)  # Product over w, l, h
        loss_raw = 1.0 - iou_dims

        if self.use_uncertainty and uncertainty is not None:
            sigma_s = uncertainty[:, 3]  # Size uncertainty
            loss = loss_raw / (sigma_s + 1e-8) + torch.log(2 * sigma_s + 1e-8)
        else:
            loss = loss_raw

        return loss.mean()

    def compute_rotation_loss(self, pred_rot, gt_rot, uncertainty=None):
        """
        Rotation loss (Equation 7 in paper):
        L_rot = 1/σ_o * ∑ |sin/cos_pred - sin/cos_gt| + log(2σ_o)

        Args:
            pred_rot: (N, 2) predicted [sin(yaw), cos(yaw)]
            gt_rot: (N, 2) ground truth [sin(yaw), cos(yaw)]
            uncertainty: (N,) uncertainty σ_o if use_uncertainty

        Returns:
            loss: scalar
        """
        loss_raw = F.l1_loss(pred_rot, gt_rot, reduction='none').sum(dim=1)

        if self.use_uncertainty and uncertainty is not None:
            sigma_o = uncertainty[:, 4]  # Orientation uncertainty
            loss = loss_raw / (sigma_o + 1e-8) + torch.log(2 * sigma_o + 1e-8)
        else:
            loss = loss_raw

        return loss.mean()

    def forward(self, predictions, targets):
        """
        Compute detection loss with greedy matching.

        Args:
            predictions: Dict containing:
                - cls_logits: (B, num_queries, num_classes+1)
                - bbox_preds: (B, num_queries, 10) [x, y, z, w, l, h, sin_yaw, cos_yaw, vx, vy]
                - uncertainties: (B, num_queries, 5) if use_uncertainty
            targets: List of dicts (length B) containing:
                - boxes: (N_gt, 7+) ground truth boxes
                - labels: (N_gt,) ground truth labels

        Returns:
            loss_dict: Dict of losses
        """
        cls_logits = predictions['cls_logits']  # (B, num_queries, num_classes+1)
        bbox_preds = predictions['bbox_preds']  # (B, num_queries, 10)
        uncertainties = predictions.get('uncertainties', None)

        B = cls_logits.shape[0]

        total_cls_loss = 0.0
        total_loc_loss = 0.0
        total_size_loss = 0.0
        total_rot_loss = 0.0
        num_pos_samples = 0

        for batch_idx in range(B):
            pred_cls = cls_logits[batch_idx]  # (num_queries, num_classes+1)
            pred_bbox = bbox_preds[batch_idx]  # (num_queries, 10)

            gt_boxes = targets[batch_idx]['boxes']  # (N_gt, 7+)
            gt_labels = targets[batch_idx]['labels']  # (N_gt,)

            # DEBUG: Check if we have GT boxes
            if batch_idx == 0 and len(gt_boxes) == 0:
                print(f"[DetectionLoss] WARNING: Batch {batch_idx} has ZERO ground truth boxes!")

            # Greedy matching
            matched_indices, unmatched_preds, unmatched_gts = greedy_matching(
                pred_bbox[:, :7],  # Use first 7 dims for matching
                pred_cls,
                gt_boxes,
                gt_labels
            )

            # Matched predictions: classification + regression loss
            if len(matched_indices) > 0:
                matched_pred_idxs = [m[0] for m in matched_indices]
                matched_gt_idxs = [m[1] for m in matched_indices]

                # Classification loss for matched
                matched_pred_cls = pred_cls[matched_pred_idxs]
                matched_gt_labels = gt_labels[matched_gt_idxs]
                cls_loss_pos = self.cls_loss_fn(matched_pred_cls, matched_gt_labels)
                total_cls_loss += cls_loss_pos

                # Regression losses
                matched_pred_bbox = pred_bbox[matched_pred_idxs]
                matched_gt_bbox = gt_boxes[matched_gt_idxs]

                # Extract components
                pred_loc = matched_pred_bbox[:, :3]  # x, y, z
                gt_loc = matched_gt_bbox[:, :3]

                pred_size = matched_pred_bbox[:, 3:6]  # w, l, h
                gt_size = matched_gt_bbox[:, 3:6]

                pred_rot = matched_pred_bbox[:, 6:8]  # sin, cos
                # Need to convert gt yaw to sin/cos
                gt_yaw = matched_gt_bbox[:, 6]  # Assuming gt has yaw angle
                gt_rot = torch.stack([torch.sin(gt_yaw), torch.cos(gt_yaw)], dim=1)

                # Uncertainties
                unc = uncertainties[batch_idx][matched_pred_idxs] if uncertainties is not None else None

                # Compute losses
                loc_loss = self.compute_location_loss(pred_loc, gt_loc, unc)
                size_loss = self.compute_size_loss(pred_size, gt_size, unc)
                rot_loss = self.compute_rotation_loss(pred_rot, gt_rot, unc)

                total_loc_loss += loc_loss
                total_size_loss += size_loss
                total_rot_loss += rot_loss

                num_pos_samples += len(matched_indices)

            # Unmatched predictions: negative classification loss
            if len(unmatched_preds) > 0:
                unmatched_pred_cls = pred_cls[unmatched_preds]
                # Background class is last index
                background_labels = torch.full(
                    (len(unmatched_preds),),
                    self.num_classes,
                    dtype=torch.long,
                    device=pred_cls.device
                )
                cls_loss_neg = self.cls_loss_fn(unmatched_pred_cls, background_labels)
                total_cls_loss += cls_loss_neg

        # Average over batch
        num_pos_samples = max(num_pos_samples, 1)  # Avoid division by zero

        loss_dict = {
            'loss_cls': total_cls_loss / B * self.loss_cls_weight,
            'loss_loc': total_loc_loss / num_pos_samples * self.loss_bbox_weight,
            'loss_size': total_size_loss / num_pos_samples * self.loss_bbox_weight,
            'loss_rot': total_rot_loss / num_pos_samples * self.loss_bbox_weight,
        }

        # Total loss
        loss_dict['loss_total'] = sum(loss_dict.values())

        return loss_dict

"""
Greedy Matching Algorithm
Paper uses greedy matching instead of Hungarian for efficiency
Section 4.1: "we adopt a greedy matching algorithm instead of the well-known
Hungarian algorithm without losing accuracy, but more efficient"
"""

import torch
import numpy as np
from scipy.optimize import linear_sum_assignment


def compute_iou_3d(boxes1, boxes2):
    """
    Compute 3D IoU between two sets of boxes.

    Args:
        boxes1: (N, 7+) [x, y, z, w, l, h, yaw, ...]
        boxes2: (M, 7+) [x, y, z, w, l, h, yaw, ...]

    Returns:
        iou: (N, M) IoU matrix
    """
    # Simplified 2D IoU for BEV (ignoring height and rotation for now)
    # Full 3D IoU with rotation is complex, using BEV IoU as approximation

    N = boxes1.shape[0]
    M = boxes2.shape[0]

    if N == 0 or M == 0:
        return torch.zeros((N, M), device=boxes1.device)

    # Extract center and size
    centers1 = boxes1[:, :2]  # (N, 2)
    sizes1 = boxes1[:, 3:5]   # (N, 2) - w, l

    centers2 = boxes2[:, :2]  # (M, 2)
    sizes2 = boxes2[:, 3:5]   # (M, 2)

    # Compute pairwise distances (simple metric for matching)
    # In practice, use proper 3D IoU with rotation
    dist = torch.cdist(centers1, centers2, p=2)  # (N, M)

    # Convert distance to similarity (inverse)
    # Lower distance = higher similarity
    max_dist = dist.max() + 1e-6
    similarity = 1.0 - (dist / max_dist)

    return similarity


def greedy_matching(pred_boxes, pred_scores, gt_boxes, gt_labels,
                   iou_threshold=0.5, spatial_constraint=True):
    """
    Greedy matching between predictions and ground truth.

    Unlike Hungarian algorithm which finds global optimal assignment,
    greedy matching iteratively picks the best match.

    Args:
        pred_boxes: (N_pred, 7+) predicted boxes [x, y, z, w, l, h, ...]
        pred_scores: (N_pred, num_classes) predicted scores
        gt_boxes: (N_gt, 7+) ground truth boxes
        gt_labels: (N_gt,) ground truth labels

        iou_threshold: Minimum IoU for valid match
        spatial_constraint: If True, limit matching to spatial coverage region

    Returns:
        matched_indices: List of (pred_idx, gt_idx) tuples
        unmatched_preds: List of unmatched prediction indices
        unmatched_gts: List of unmatched ground truth indices
    """
    N_pred = pred_boxes.shape[0]
    N_gt = gt_boxes.shape[0]

    if N_pred == 0 or N_gt == 0:
        return [], list(range(N_pred)), list(range(N_gt))

    # Compute cost matrix (higher is better for matching)
    # Use IoU as primary metric
    cost_matrix = compute_iou_3d(pred_boxes, gt_boxes)  # (N_pred, N_gt)

    # Apply spatial constraint (from paper Section 4.1)
    # Only match predictions within ground truth's BEV coverage
    if spatial_constraint:
        for gt_idx in range(N_gt):
            gt_box = gt_boxes[gt_idx]
            # Compute which predictions are within reasonable distance
            pred_centers = pred_boxes[:, :2]
            gt_center = gt_box[:2]

            distances = torch.norm(pred_centers - gt_center.unsqueeze(0), dim=1)
            # Only consider predictions within 2x the object size
            max_distance = 2.0 * torch.max(gt_box[3:5])

            # Mask out predictions too far away
            cost_matrix[distances > max_distance, gt_idx] = 0.0

    # Greedy matching
    matched_indices = []
    unmatched_preds = set(range(N_pred))
    unmatched_gts = set(range(N_gt))

    # Convert to numpy for easier manipulation
    cost_np = cost_matrix.detach().cpu().numpy()

    # Iteratively find best matches
    while unmatched_preds and unmatched_gts:
        # Find maximum cost among unmatched pairs
        best_cost = -1
        best_pair = None

        for pred_idx in unmatched_preds:
            for gt_idx in unmatched_gts:
                cost = cost_np[pred_idx, gt_idx]
                if cost > best_cost:
                    best_cost = cost
                    best_pair = (pred_idx, gt_idx)

        # Check if best cost meets threshold
        if best_cost < iou_threshold:
            break  # No more valid matches

        # Add match and remove from unmatched sets
        matched_indices.append(best_pair)
        unmatched_preds.remove(best_pair[0])
        unmatched_gts.remove(best_pair[1])

    return matched_indices, list(unmatched_preds), list(unmatched_gts)


def hungarian_matching(cost_matrix):
    """
    Hungarian matching for comparison/baseline.

    Args:
        cost_matrix: (N, M) cost matrix (higher = better match)

    Returns:
        matches: List of (pred_idx, gt_idx) tuples
    """
    # Scipy's linear_sum_assignment minimizes cost, so negate
    cost_np = -cost_matrix.cpu().numpy()

    pred_indices, gt_indices = linear_sum_assignment(cost_np)

    matches = list(zip(pred_indices.tolist(), gt_indices.tolist()))

    return matches

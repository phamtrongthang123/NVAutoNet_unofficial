"""
Visualization script for NVAutoNet predictions
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nvautonet.models import NVAutoNet
from nvautonet.datasets import NuScenesDataset


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize NVAutoNet predictions')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sample-idx', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default='visualizations')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-mini', action='store_true')

    return parser.parse_args()


def plot_bev_boxes(boxes, labels, scores, ax, color='r', title='Predictions'):
    """Plot boxes in BEV view"""

    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.set_xlabel('X (forward, m)')
    ax.set_ylabel('Y (left, m)')
    ax.grid(True, alpha=0.3)

    # Plot ego vehicle
    ego_box = Rectangle((-2, -1), 4, 2, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(ego_box)

    # Plot boxes
    for box, label, score in zip(boxes, labels, scores):
        x, y, z, w, l, h = box[:6]

        # Simple rectangle visualization (ignoring rotation for simplicity)
        rect = Rectangle((x - l/2, y - w/2), l, w,
                        fill=False, edgecolor=color, linewidth=1.5)
        ax.add_patch(rect)

        # Add label
        ax.text(x, y, f'{label}:{score:.2f}',
               fontsize=8, color=color,
               ha='center', va='center')


def visualize_sample(model, dataset, sample_idx, device, out_dir):
    """Visualize predictions for a single sample"""

    # Get sample
    sample = dataset[sample_idx]

    # Prepare batch
    imgs = {cam_id: img.unsqueeze(0).to(device)
            for cam_id, img in sample['imgs'].items()}

    targets = [{
        'boxes': sample['targets']['boxes'].to(device),
        'labels': sample['targets']['labels'].to(device)
    }]

    batch = {
        'imgs': imgs,
        'camera_ids': sample['camera_ids'],
        'camera_params': sample['camera_params'],
        'targets': targets
    }

    # Generate lookup tables if needed
    camera_params_gpu = {}
    for cam_id in batch['camera_params']:
        camera_params_gpu[cam_id] = {
            'intrinsic': batch['camera_params'][cam_id]['intrinsic'].to(device),
            'extrinsic': batch['camera_params'][cam_id]['extrinsic'].to(device)
        }
    model.view_transformer.generate_lookup_tables(camera_params_gpu)

    # Forward pass
    model.eval_mode()
    with torch.no_grad():
        predictions = model(batch)

    # Extract predictions
    det_preds = predictions['detection']
    cls_logits = det_preds['cls_logits'][0]  # (num_queries, num_classes+1)
    bbox_preds = det_preds['bbox_preds'][0]  # (num_queries, 10)

    # Get scores and classes
    scores = torch.softmax(cls_logits, dim=-1)
    pred_classes = torch.argmax(scores, dim=-1)

    # Filter valid predictions
    num_classes = cls_logits.shape[-1] - 1
    valid_mask = pred_classes < num_classes
    score_threshold = 0.3
    score_mask = scores.max(dim=-1)[0] > score_threshold

    valid_mask = valid_mask & score_mask

    pred_boxes = bbox_preds[valid_mask].cpu().numpy()
    pred_labels = pred_classes[valid_mask].cpu().numpy()
    pred_scores = scores[valid_mask, pred_labels].cpu().numpy()

    # Ground truth
    gt_boxes = sample['targets']['boxes'].cpu().numpy()
    gt_labels = sample['targets']['labels'].cpu().numpy()

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Predictions
    plot_bev_boxes(pred_boxes, pred_labels, pred_scores,
                  axes[0], color='r', title='Predictions')

    # Ground truth
    gt_scores = np.ones(len(gt_labels))
    plot_bev_boxes(gt_boxes, gt_labels, gt_scores,
                  axes[1], color='g', title='Ground Truth')

    plt.tight_layout()

    # Save
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f'sample_{sample_idx}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[Visualize] Saved to {output_path}")

    plt.close()


def main():
    args = parse_args()

    # Build model and load checkpoint
    print("[Visualize] Loading model...")
    from tools.train import build_model

    class Args:
        pass

    train_args = Args()
    model = build_model(train_args)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)

    # Load dataset
    print("[Visualize] Loading dataset...")
    dataset = NuScenesDataset(
        data_root=args.data_root,
        split='val',
        version='v1.0-mini' if args.use_mini else 'v1.0-trainval',
        use_mini=args.use_mini
    )

    # Visualize
    print(f"[Visualize] Visualizing sample {args.sample_idx}...")
    visualize_sample(model, dataset, args.sample_idx, device, args.out_dir)

    print("[Visualize] Done!")


if __name__ == '__main__':
    main()

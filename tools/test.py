"""
Evaluation script for NVAutoNet
"""

import os
import sys
import argparse
import torch
import numpy as np
from tqdm import tqdm
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nvautonet.models import NVAutoNet
from nvautonet.datasets import NuScenesDataset, collate_fn
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate NVAutoNet')
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use-mini', action='store_true')
    parser.add_argument('--save-results', type=str, default=None)

    return parser.parse_args()


def evaluate(model, dataloader, device):
    """Run evaluation"""

    model.eval_mode()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation'):
            # Move to device
            for cam_id in batch['imgs']:
                batch['imgs'][cam_id] = batch['imgs'][cam_id].to(device)

            # Forward pass
            predictions = model(batch)

            # Store predictions and targets
            det_preds = predictions['detection']

            # Process predictions for each sample in batch
            B = det_preds['cls_logits'].shape[0]

            for b in range(B):
                cls_logits = det_preds['cls_logits'][b]  # (num_queries, num_classes+1)
                bbox_preds = det_preds['bbox_preds'][b]  # (num_queries, 10)

                # Get scores and predicted classes
                scores = torch.softmax(cls_logits, dim=-1)
                pred_classes = torch.argmax(scores, dim=-1)

                # Filter out background (class = num_classes)
                num_classes = cls_logits.shape[-1] - 1
                valid_mask = pred_classes < num_classes

                valid_boxes = bbox_preds[valid_mask]
                valid_classes = pred_classes[valid_mask]
                valid_scores = scores[valid_mask, valid_classes]

                # Store
                all_predictions.append({
                    'boxes': valid_boxes.cpu().numpy(),
                    'labels': valid_classes.cpu().numpy(),
                    'scores': valid_scores.cpu().numpy(),
                    'sample_token': batch['sample_tokens'][b]
                })

                all_targets.append({
                    'boxes': batch['targets'][b]['boxes'].cpu().numpy(),
                    'labels': batch['targets'][b]['labels'].cpu().numpy(),
                    'sample_token': batch['sample_tokens'][b]
                })

    return all_predictions, all_targets


def compute_metrics(predictions, targets):
    """Compute evaluation metrics"""

    # Simple precision/recall calculation
    num_tp = 0
    num_fp = 0
    num_gt = 0

    for pred, target in zip(predictions, targets):
        num_gt += len(target['boxes'])

        # Simple matching based on class
        pred_labels = pred['labels']
        gt_labels = target['labels']

        for pred_label in pred_labels:
            if pred_label in gt_labels:
                num_tp += 1
            else:
                num_fp += 1

    precision = num_tp / (num_tp + num_fp + 1e-8)
    recall = num_tp / (num_gt + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_predictions': num_tp + num_fp,
        'num_ground_truth': num_gt
    }

    return metrics


def main():
    args = parse_args()

    # Build model
    print("[Test] Loading model...")
    from tools.train import build_model

    class Args:
        pass

    train_args = Args()
    model = build_model(train_args)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])

    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)

    print(f"[Test] Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Build dataloader
    print("[Test] Loading dataset...")
    dataset = NuScenesDataset(
        data_root=args.data_root,
        split='val',
        version='v1.0-mini' if args.use_mini else 'v1.0-trainval',
        use_mini=args.use_mini
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Generate BEV lookup tables
    print("[Test] Generating BEV lookup tables...")
    sample_batch = next(iter(dataloader))
    camera_params = sample_batch['camera_params']
    camera_params_gpu = {}
    for cam_id in camera_params:
        camera_params_gpu[cam_id] = {
            'intrinsic': camera_params[cam_id]['intrinsic'].to(device),
            'extrinsic': camera_params[cam_id]['extrinsic'].to(device)
        }
    model.view_transformer.generate_lookup_tables(camera_params_gpu)

    # Run evaluation
    print("[Test] Running evaluation...")
    predictions, targets = evaluate(model, dataloader, device)

    # Compute metrics
    print("[Test] Computing metrics...")
    metrics = compute_metrics(predictions, targets)

    print("\n" + "="*60)
    print("Evaluation Results")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("="*60 + "\n")

    # Save results
    if args.save_results:
        results = {
            'metrics': metrics,
            'predictions': [
                {k: v.tolist() if isinstance(v, np.ndarray) else v
                 for k, v in pred.items()}
                for pred in predictions
            ]
        }

        os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[Test] Results saved to {args.save_results}")


if __name__ == '__main__':
    main()

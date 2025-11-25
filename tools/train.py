"""
Training script for NVAutoNet
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nvautonet.models import NVAutoNet
from nvautonet.datasets import NuScenesDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description='Train NVAutoNet')
    parser.add_argument('--data-root', type=str, required=True,
                       help='Path to nuScenes dataset')
    parser.add_argument('--work-dir', type=str, default='work_dirs/nvautonet',
                       help='Directory to save checkpoints and logs')
    parser.add_argument('--batch-size', type=int, default=2,
                       help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=24,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--use-mini', action='store_true',
                       help='Use mini split for quick testing')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU id to use')

    return parser.parse_args()


def build_model(args):
    """Build NVAutoNet model"""

    model_cfg = {
        'img_backbone': 'resnet50',
        'img_backbone_pretrained': True,
        'img_feat_channels': 2048,
        'view_transformer_cfg': {
            'in_channels': 2048,
            'out_channels': 80,
            'image_size': (15, 25),  # After 32x downsampling: 450/32=15, 800/32=25
            'bev_h': 200,
            'bev_w': 200,
            'x_bound': [-50.0, 50.0, 0.5],
            'y_bound': [-50.0, 50.0, 0.5],
            'mlp_hidden_dims': [256, 128],
            'num_depth_bins': 64,
        },
        'bev_encoder_cfg': {
            'in_channels': 80,
            'block_channels': [64, 128, 256],
            'out_indices': (2,),  # Only return last block output
        },
        'detection_head_cfg': {
            'num_classes': 10,  # nuScenes has 10 classes
            'num_queries': 300,
            'embed_dims': 256,
            'bev_channels': 256,
        },
        'segmentation_head_cfg': None,  # Disabled for initial training
        'use_adaptive_loss_balancing': True,
        'task_priors': {'detection': 5.0}  # Higher weight for detection
    }

    model = NVAutoNet(**model_cfg)

    # Generate BEV lookup tables from first sample
    # In practice, these should be precomputed for each camera configuration
    print("[Train] Note: BEV lookup tables will be generated on first forward pass")

    return model


def build_dataloader(args, split='train'):
    """Build data loader"""

    dataset = NuScenesDataset(
        data_root=args.data_root,
        split=split,
        version='v1.0-mini' if args.use_mini else 'v1.0-trainval',
        use_mini=args.use_mini
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(split == 'train'),
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataloader


def train_one_epoch(model, dataloader, optimizer, scheduler, epoch, writer, args):
    """Train for one epoch"""

    model.train_mode()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(pbar):
        # Move to GPU
        device = torch.device(f'cuda:{args.gpu}')

        # Move images to GPU
        for cam_id in batch['imgs']:
            batch['imgs'][cam_id] = batch['imgs'][cam_id].to(device)

        # Move targets to GPU
        for target in batch['targets']:
            target['boxes'] = target['boxes'].to(device)
            target['labels'] = target['labels'].to(device)

        # Generate BEV lookup tables (only once, on first batch)
        if batch_idx == 0 and epoch == 0:
            print("[Train] Generating BEV lookup tables...")
            camera_params = batch['camera_params']
            camera_params_gpu = {}
            for cam_id in camera_params:
                camera_params_gpu[cam_id] = {
                    'intrinsic': camera_params[cam_id]['intrinsic'].to(device),
                    'extrinsic': camera_params[cam_id]['extrinsic'].to(device)
                }
            model.view_transformer.generate_lookup_tables(camera_params_gpu)
            print("[Train] BEV lookup tables generated")

        # Forward pass
        loss_dict = model(batch)

        total_loss = loss_dict['total_weighted']

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=35.0)

        optimizer.step()

        # Logging
        global_step = epoch * len(dataloader) + batch_idx

        if batch_idx % 10 == 0:
            for key, value in loss_dict.items():
                if isinstance(value, torch.Tensor):
                    writer.add_scalar(f'train/{key}', value.item(), global_step)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

    scheduler.step()

    # Update loss balancer weights at end of epoch
    model.loss_balancer.update_weights()
    model.loss_balancer.reset_epoch_losses()


def validate(model, dataloader, epoch, writer, args):
    """Validation"""

    model.eval()  # Set PyTorch eval mode
    # But keep training_mode=True so losses are computed
    # We just won't accumulate them for loss balancing
    model.training_mode = True

    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Move to GPU
            device = torch.device(f'cuda:{args.gpu}')

            for cam_id in batch['imgs']:
                batch['imgs'][cam_id] = batch['imgs'][cam_id].to(device)

            for target in batch['targets']:
                target['boxes'] = target['boxes'].to(device)
                target['labels'] = target['labels'].to(device)

            # Forward pass - will compute losses since training_mode=True
            loss_dict = model(batch)

            total_loss += loss_dict['total_weighted'].item()
            num_batches += 1

    avg_loss = total_loss / num_batches

    print(f'\n[Validation] Epoch {epoch}: Avg Loss = {avg_loss:.4f}\n')

    writer.add_scalar('val/loss', avg_loss, epoch)

    # Reset to training mode
    model.train()
    model.training_mode = True

    return avg_loss


def main():
    args = parse_args()

    # Create work directory
    os.makedirs(args.work_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(os.path.join(args.work_dir, 'tensorboard'))

    # Build model
    print("[Train] Building model...")
    model = build_model(args)

    # Move to GPU
    device = torch.device(f'cuda:{args.gpu}')
    model = model.to(device)

    print(f"[Train] Model has {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    # Build dataloaders
    print("[Train] Building dataloaders...")
    train_dataloader = build_dataloader(args, split='train')
    val_dataloader = build_dataloader(args, split='val')

    print(f"[Train] Train samples: {len(train_dataloader.dataset)}")
    print(f"[Train] Val samples: {len(val_dataloader.dataset)}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=2e-6
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f"[Train] Resuming from {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f'\n{"="*60}')
        print(f'Epoch {epoch}/{args.epochs}')
        print(f'{"="*60}\n')

        # Train
        train_one_epoch(model, train_dataloader, optimizer, scheduler, epoch, writer, args)

        # Validate
        val_loss = validate(model, val_dataloader, epoch, writer, args)

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_loss': val_loss,
            'loss_balancer': model.loss_balancer.state_dict()
        }

        # Save latest
        torch.save(checkpoint, os.path.join(args.work_dir, 'latest.pth'))

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(args.work_dir, 'best.pth'))
            print(f'[Train] Saved best checkpoint (val_loss={val_loss:.4f})')

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(checkpoint, os.path.join(args.work_dir, f'epoch_{epoch}.pth'))

    print(f'\n[Train] Training completed! Best val loss: {best_val_loss:.4f}')
    writer.close()


if __name__ == '__main__':
    main()

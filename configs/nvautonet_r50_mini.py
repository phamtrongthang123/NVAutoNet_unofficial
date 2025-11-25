"""
Configuration for NVAutoNet with ResNet50 on nuScenes mini split
"""

# Model configuration
model = dict(
    type='NVAutoNet',
    img_backbone='resnet50',
    img_backbone_pretrained=True,
    img_feat_channels=2048,

    # Column-wise MLP View Transformer
    view_transformer_cfg=dict(
        in_channels=2048,
        out_channels=80,
        image_size=(15, 25),  # After 32x downsampling: 450/32=15, 800/32=25
        bev_h=200,
        bev_w=200,
        x_bound=[-50.0, 50.0, 0.5],  # X: -50m to +50m, 0.5m resolution
        y_bound=[-50.0, 50.0, 0.5],  # Y: -50m to +50m, 0.5m resolution
        mlp_hidden_dims=[256, 128],
        num_depth_bins=64,
        use_depth_supervision=False
    ),

    # BEV Encoder
    bev_encoder_cfg=dict(
        in_channels=80,
        block_channels=[64, 128, 256],
        block_strides=[1, 2, 2],
        block_repeats=[4, 4, 4],
        kernel_size=3,
        out_indices=(2,)  # Only return last block output
    ),

    # Detection Head
    detection_head_cfg=dict(
        num_classes=10,  # nuScenes: 10 classes
        num_queries=300,
        embed_dims=256,
        bev_channels=256,
        num_reg_fcs=2,
        use_uncertainty=True
    ),

    # Segmentation Head (disabled for mini split)
    segmentation_head_cfg=None,

    # Loss configuration
    loss_cfg=dict(
        detection=dict(
            num_classes=10,
            loss_cls_weight=1.0,
            loss_bbox_weight=1.0,
            use_focal_loss=True,
            use_uncertainty=True
        )
    ),

    # Multi-task learning
    use_adaptive_loss_balancing=True,
    task_priors=dict(
        detection=5.0  # Higher priority for detection
    )
)

# Dataset configuration
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
version = 'v1.0-mini'

# Training configuration
train_cfg = dict(
    batch_size=2,  # Per GPU
    num_epochs=24,
    lr=2e-4,
    weight_decay=0.01,
    gradient_clip=35.0,
    use_mini=True
)

# Optimizer
optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.01
)

# Learning rate scheduler
lr_config = dict(
    type='CosineAnnealing',
    T_max=24,
    eta_min=2e-6
)

# Logging
log_config = dict(
    interval=10,
    tensorboard_dir='work_dirs/nvautonet_mini/tensorboard'
)

# Checkpointing
checkpoint_config = dict(
    interval=5,  # Save every 5 epochs
    save_best=True,
    metric='val_loss',
    mode='min'
)

# Work directory
work_dir = 'work_dirs/nvautonet_mini'

# GPU
gpu_ids = [0]

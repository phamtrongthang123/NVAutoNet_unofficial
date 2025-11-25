#!/bin/bash
# Test nuScenes dataloader
# Usage: bash scripts/test_dataloader.sh

set -e

echo "=================================="
echo "Testing nuScenes Dataloader"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if data exists
DATA_DIR="data/nuscenes"

if [ ! -d "${DATA_DIR}/v1.0-mini" ]; then
    echo -e "${RED}Error: nuScenes mini dataset not found!${NC}"
    echo "Please run: bash scripts/download_data.sh"
    exit 1
fi

echo -e "${GREEN}✓ Found nuScenes dataset${NC}"
echo ""

# Test dataloader
cd "$(dirname "$0")/.."

python << 'EOF'
import sys
sys.path.insert(0, '.')
import torch
from nvautonet.datasets import NuScenesDataset, collate_fn
from torch.utils.data import DataLoader

print("1. Creating dataset...")
dataset = NuScenesDataset(
    data_root='data/nuscenes',
    split='train',
    version='v1.0-mini',
    use_mini=True
)

print(f"   ✓ Loaded {len(dataset)} training samples")
print()

print("2. Testing single sample...")
sample = dataset[0]
print(f"   ✓ Sample keys: {list(sample.keys())}")
print(f"   ✓ Number of cameras: {len(sample['camera_ids'])}")
print(f"   ✓ Camera IDs: {sample['camera_ids']}")

for cam_id in sample['camera_ids']:
    img = sample['imgs'][cam_id]
    print(f"   ✓ {cam_id} shape: {img.shape}")

print(f"   ✓ Number of GT boxes: {len(sample['targets']['boxes'])}")
print(f"   ✓ GT box shape: {sample['targets']['boxes'].shape}")
print(f"   ✓ GT labels: {sample['targets']['labels']}")
print()

print("3. Testing dataloader with batching...")
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,  # Use 0 for testing
    collate_fn=collate_fn,
    drop_last=True
)

batch = next(iter(dataloader))
print(f"   ✓ Batch keys: {list(batch.keys())}")
print(f"   ✓ Batch size: {batch['imgs'][batch['camera_ids'][0]].shape[0]}")

for cam_id in batch['camera_ids']:
    img_batch = batch['imgs'][cam_id]
    print(f"   ✓ {cam_id} batch shape: {img_batch.shape}")

print(f"   ✓ Number of samples in batch: {len(batch['targets'])}")
print(f"   ✓ Camera parameters available: {list(batch['camera_params'].keys())}")
print()

print("4. Iterating through few batches...")
for i, batch in enumerate(dataloader):
    if i >= 3:
        break
    print(f"   ✓ Batch {i+1}: {len(batch['targets'])} samples")

print()
print("="*50)
print("✓ All dataloader tests passed!")
print("="*50)
print()
print("Dataset is ready for training.")
EOF

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}Dataloader test successful! ✓${NC}"
    echo ""
    echo "Next step: Run quick training test"
    echo "  bash scripts/train_quick.sh"
else
    echo ""
    echo -e "${RED}Dataloader test failed!${NC}"
    exit 1
fi

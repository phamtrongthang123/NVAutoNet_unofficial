#!/bin/bash
# Quick training test (3 epochs for debugging)
# Usage: bash scripts/train_quick.sh

set -e

echo "=================================="
echo "Quick Training Test (3 epochs)"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check data
if [ ! -d "data/nuscenes/v1.0-mini" ]; then
    echo "Error: nuScenes data not found!"
    echo "Run: bash scripts/download_data.sh"
    exit 1
fi

# Training config
WORK_DIR="work_dirs/quick_test"
BATCH_SIZE=1  # Small batch for quick testing
EPOCHS=3      # Just 3 epochs for quick validation
GPU=0

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU available${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}Warning: No GPU found, training will be slow${NC}"
    echo ""
fi

# Create work directory
mkdir -p ${WORK_DIR}

# Run training
echo "Starting training..."
echo "  Work dir: ${WORK_DIR}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Epochs: ${EPOCHS}"
echo "  GPU: ${GPU}"
echo ""
echo "This will take approximately 5-10 minutes..."
echo ""

python tools/train.py \
    --data-root data/nuscenes \
    --work-dir ${WORK_DIR} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --lr 2e-4 \
    --use-mini \
    --gpu ${GPU} 2>&1 | tee ${WORK_DIR}/train.log

# Check if training completed successfully
if [ -f "${WORK_DIR}/latest.pth" ]; then
    echo ""
    echo "=================================="
    echo -e "${GREEN}Training completed successfully! ✓${NC}"
    echo "=================================="
    echo ""
    echo "Checkpoint saved: ${WORK_DIR}/latest.pth"
    echo ""
    echo "View logs:"
    echo "  tensorboard --logdir ${WORK_DIR}/tensorboard"
    echo ""
    echo "Next steps:"
    echo "  1. Run evaluation: bash scripts/eval_quick.sh"
    echo "  2. Visualize results: bash scripts/visualize.sh"
    echo "  3. Full training: bash scripts/train_full.sh"
else
    echo ""
    echo "Error: Training failed!"
    echo "Check logs: ${WORK_DIR}/train.log"
    exit 1
fi

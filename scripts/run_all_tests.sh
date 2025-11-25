#!/bin/bash
# Run complete test pipeline (setup -> data -> train -> eval -> visualize)
# Usage: bash scripts/run_all_tests.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "=========================================="
echo "NVAutoNet Complete Test Pipeline"
echo "=========================================="
echo ""
echo "This will:"
echo "  1. Setup environment (if needed)"
echo "  2. Download nuScenes mini data (if needed)"
echo "  3. Test installation"
echo "  4. Test dataloader"
echo "  5. Quick training (3 epochs)"
echo "  6. Evaluation"
echo "  7. Visualization"
echo ""
echo -e "${YELLOW}Total estimated time: 15-20 minutes${NC}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Change to project root
cd "$(dirname "$0")/.."

# Step 1: Check environment
echo ""
echo "=========================================="
echo -e "${BLUE}Step 1/7: Checking Environment${NC}"
echo "=========================================="
echo ""

if ! conda env list | grep -q "^nvautonet "; then
    echo "Environment not found. Creating..."
    bash scripts/setup_env.sh
else
    echo -e "${GREEN}✓ Environment exists${NC}"

    # Activate environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate nvautonet
fi

# Step 2: Download data
echo ""
echo "=========================================="
echo -e "${BLUE}Step 2/7: Checking Dataset${NC}"
echo "=========================================="
echo ""

if [ ! -d "data/nuscenes/v1.0-mini" ]; then
    echo "Dataset not found. Downloading..."
    bash scripts/download_data.sh
else
    echo -e "${GREEN}✓ Dataset exists${NC}"
fi

# Step 3: Test installation
echo ""
echo "=========================================="
echo -e "${BLUE}Step 3/7: Testing Installation${NC}"
echo "=========================================="
echo ""

bash scripts/test_installation.sh

# Step 4: Test dataloader
echo ""
echo "=========================================="
echo -e "${BLUE}Step 4/7: Testing Dataloader${NC}"
echo "=========================================="
echo ""

bash scripts/test_dataloader.sh

# Step 5: Quick training
echo ""
echo "=========================================="
echo -e "${BLUE}Step 5/7: Quick Training (3 epochs)${NC}"
echo "=========================================="
echo ""

bash scripts/train_quick.sh

# Step 6: Evaluation
echo ""
echo "=========================================="
echo -e "${BLUE}Step 6/7: Evaluation${NC}"
echo "=========================================="
echo ""

bash scripts/eval_quick.sh

# Step 7: Visualization
echo ""
echo "=========================================="
echo -e "${BLUE}Step 7/7: Visualization${NC}"
echo "=========================================="
echo ""

bash scripts/visualize.sh 0

# Final summary
echo ""
echo "=========================================="
echo -e "${GREEN}All Tests Completed Successfully! ✓${NC}"
echo "=========================================="
echo ""
echo "Summary of results:"
echo "  - Environment: conda env 'nvautonet'"
echo "  - Dataset: data/nuscenes"
echo "  - Model checkpoint: work_dirs/quick_test/latest.pth"
echo "  - Evaluation results: work_dirs/quick_test/results.json"
echo "  - Visualizations: work_dirs/quick_test/visualizations/"
echo ""
echo "View training logs:"
echo "  tensorboard --logdir work_dirs/quick_test/tensorboard"
echo ""
echo "Run full training (24 epochs):"
echo "  bash scripts/train_full.sh"
echo ""
echo -e "${GREEN}Everything is working correctly!${NC}"

#!/bin/bash
# Quick evaluation test
# Usage: bash scripts/eval_quick.sh

set -e

echo "=================================="
echo "Quick Evaluation Test"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Config
WORK_DIR="work_dirs/quick_test"
CHECKPOINT="${WORK_DIR}/latest.pth"
RESULTS_FILE="${WORK_DIR}/results.json"

# Check if checkpoint exists
if [ ! -f "${CHECKPOINT}" ]; then
    echo -e "${RED}Error: Checkpoint not found: ${CHECKPOINT}${NC}"
    echo "Please run training first: bash scripts/train_quick.sh"
    exit 1
fi

echo -e "${GREEN}✓ Found checkpoint: ${CHECKPOINT}${NC}"
echo ""

# Run evaluation
echo "Running evaluation..."
echo ""

python tools/test.py \
    --data-root data/nuscenes \
    --checkpoint ${CHECKPOINT} \
    --batch-size 1 \
    --use-mini \
    --save-results ${RESULTS_FILE} \
    --gpu 0

# Check results
if [ -f "${RESULTS_FILE}" ]; then
    echo ""
    echo "=================================="
    echo -e "${GREEN}Evaluation completed! ✓${NC}"
    echo "=================================="
    echo ""
    echo "Results saved: ${RESULTS_FILE}"
    echo ""

    # Display metrics
    echo "Metrics:"
    python -c "import json; metrics = json.load(open('${RESULTS_FILE}'))['metrics']; [print(f'  {k}: {v:.4f}') for k, v in metrics.items()]"

    echo ""
    echo "Next step: Visualize predictions"
    echo "  bash scripts/visualize.sh"
else
    echo ""
    echo -e "${RED}Evaluation failed!${NC}"
    exit 1
fi

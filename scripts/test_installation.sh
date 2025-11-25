#!/bin/bash
# Test NVAutoNet installation
# Usage: bash scripts/test_installation.sh

# Note: Don't use 'set -e' here as we want to track all test failures
# and show summary at the end

echo "=================================="
echo "NVAutoNet Installation Test"
echo "=================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Track test results
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run test
run_test() {
    local test_name=$1
    local test_command=$2

    echo -n "Testing ${test_name}... "

    if eval ${test_command} > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASS${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# Test 1: Python imports
echo "1. Testing Python imports..."
run_test "torch import" "python -c 'import torch'"
run_test "torchvision import" "python -c 'import torchvision'"
run_test "numpy import" "python -c 'import numpy'"
run_test "matplotlib import" "python -c 'import matplotlib'"
run_test "nuScenes import" "python -c 'from nuscenes.nuscenes import NuScenes'"
echo ""

# Test 2: NVAutoNet modules
echo "2. Testing NVAutoNet modules..."
cd "$(dirname "$0")/.."

run_test "nvautonet.utils" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.utils import generate_bev_grid'"
run_test "nvautonet.models.necks" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.models.necks import ColumnwiseMLPTransformer'"
run_test "nvautonet.models.backbones" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.models.backbones import BEVResNet'"
run_test "nvautonet.models.heads" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.models.heads import NVAutoNetDetectionHead'"
run_test "nvautonet.models" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.models import NVAutoNet'"
run_test "nvautonet.core.loss" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.core.loss import DetectionLoss'"
run_test "nvautonet.core.matching" "python -c 'import sys; sys.path.insert(0, \".\"); from nvautonet.core.matching import greedy_matching'"
echo ""

# Test 3: CUDA availability
echo "3. Testing CUDA..."
python -c "import torch; cuda_available = torch.cuda.is_available(); print(f'CUDA available: {cuda_available}'); print(f'CUDA devices: {torch.cuda.device_count()}' if cuda_available else 'CPU only')"
echo ""

# Test 4: Run unit tests if available
echo "4. Running unit tests (if pytest available)..."
if command -v pytest &> /dev/null; then
    if [ -f "tests/test_components.py" ]; then
        pytest tests/test_components.py -v --tb=short || true
    else
        echo -e "${YELLOW}No test files found${NC}"
    fi
else
    echo -e "${YELLOW}pytest not installed, skipping unit tests${NC}"
fi
echo ""

# Test 5: Quick model instantiation test
echo "5. Testing model instantiation..."
python << 'EOF'
import sys
sys.path.insert(0, '.')
import torch
from nvautonet.models import NVAutoNet

print("Creating NVAutoNet model...")
model = NVAutoNet(
    img_backbone='resnet50',
    img_backbone_pretrained=False,
    view_transformer_cfg={
        'in_channels': 2048,
        'out_channels': 80,
        'image_size': (28, 50),
        'bev_h': 100,
        'bev_w': 100,
        'x_bound': [-25.0, 25.0, 0.5],
        'y_bound': [-25.0, 25.0, 0.5],
    },
    bev_encoder_cfg={
        'in_channels': 80,
        'block_channels': [32, 64, 128],
    },
    detection_head_cfg={
        'num_classes': 10,
        'num_queries': 100,
        'bev_channels': 128,
    }
)

num_params = sum(p.numel() for p in model.parameters())
print(f"✓ Model created successfully!")
print(f"  Total parameters: {num_params/1e6:.2f}M")
EOF

if [ $? -eq 0 ]; then
    ((TESTS_PASSED++))
else
    ((TESTS_FAILED++))
fi

echo ""

# Summary
echo "=================================="
echo "Test Summary"
echo "=================================="
echo -e "Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo -e "Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ ${TESTS_FAILED} -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo ""
    echo "Installation is working correctly."
    echo ""
    echo "Next steps:"
    echo "  1. Download data: bash scripts/download_data.sh"
    echo "  2. Test dataloader: bash scripts/test_dataloader.sh"
    echo "  3. Quick training: bash scripts/train_quick.sh"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    echo ""
    echo "Please check the error messages above."
    exit 1
fi

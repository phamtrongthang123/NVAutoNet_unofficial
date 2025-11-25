#!/bin/bash
# Setup environment for NVAutoNet
# Usage: bash scripts/setup_env.sh

set -e  # Exit on error

echo "=================================="
echo "NVAutoNet Environment Setup"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo -e "${RED}Error: conda not found. Please install Miniconda or Anaconda first.${NC}"
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo -e "${GREEN}✓ Found conda${NC}"

# Environment name
ENV_NAME="nvautonet"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo -e "${YELLOW}Environment '${ENV_NAME}' already exists.${NC}"
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Skipping environment creation."
        exit 0
    fi
fi

# Step 1: Create conda environment
echo ""
echo "Step 1/4: Creating conda environment with Python 3.8..."
conda create -n ${ENV_NAME} python=3.8 -y

# Activate environment
echo ""
echo "Step 2/4: Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${ENV_NAME}

# Step 2: Install PyTorch
echo ""
echo "Step 3/4: Installing PyTorch with CUDA support..."
echo "Checking for CUDA..."

if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
    echo -e "${GREEN}✓ Found CUDA ${CUDA_VERSION}${NC}"

    # Install PyTorch based on CUDA version
    if [[ "$CUDA_VERSION" == "11.3" ]] || [[ "$CUDA_VERSION" == "11.6" ]] || [[ "$CUDA_VERSION" == "11.7" ]]; then
        echo "Installing PyTorch for CUDA 11.3..."
        conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -y
    elif [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "Installing PyTorch for CUDA 12.1..."
        conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y
    else
        echo -e "${YELLOW}Warning: Unsupported CUDA version. Installing PyTorch for CUDA 11.3...${NC}"
        conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -y
    fi
else
    echo -e "${YELLOW}Warning: CUDA not found. Installing CPU-only PyTorch...${NC}"
    conda install pytorch torchvision cpuonly -c pytorch -y
fi

# Step 3: Install uv and other packages
echo ""
echo "Step 4/4: Installing dependencies with uv..."

# Install uv
pip install uv

# Install all dependencies
uv pip install -r requirements.txt

# Verify installation
echo ""
echo "=================================="
echo "Verifying Installation"
echo "=================================="

# Check Python
PYTHON_VERSION=$(python --version)
echo -e "${GREEN}✓ ${PYTHON_VERSION}${NC}"

# Check PyTorch
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" || echo -e "${RED}✗ PyTorch import failed${NC}"

# Check CUDA
python -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" || echo -e "${RED}✗ CUDA check failed${NC}"

# Check nuScenes
python -c "from nuscenes.nuscenes import NuScenes; print('✓ nuScenes devkit')" || echo -e "${RED}✗ nuScenes import failed${NC}"

# Check NVAutoNet
cd "$(dirname "$0")/.."
python -c "import sys; sys.path.insert(0, '.'); from nvautonet.models import NVAutoNet; print('✓ NVAutoNet package')" || echo -e "${RED}✗ NVAutoNet import failed${NC}"

echo ""
echo "=================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "Next steps:"
echo "  1. Download nuScenes data: bash scripts/download_data.sh"
echo "  2. Run quick test: bash scripts/test_installation.sh"
echo "  3. Train model: bash scripts/train_quick.sh"

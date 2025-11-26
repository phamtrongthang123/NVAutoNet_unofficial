# NVAutoNet (Work in progress) 

Unofficial PyTorch implementation of [NVAutoNet](https://arxiv.org/abs/2303.12976v4) for nuScenes.

**What's implemented:** Column-wise MLP transformer, greedy matching, decomposed losses (Eq 5-7), adaptive loss balancing (Eq 14)

**What's different:** ResNet50 backbone (not custom NAS), Cartesian BEV 200×200/50m (not Polar 360×64/200m), nuScenes 6-cam (not 8-cam fisheye). Performance: 10-15 FPS vs paper's 53 FPS, mAP ~0.40 vs 0.465.

![](https://img.shields.io/badge/status-research-yellow) Not affiliated with NVIDIA.

## install

```bash
conda create -n nvautonet python=3.8 -y
conda activate nvautonet
conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
```

## run

```bash
# download nuScenes mini (4GB)
bash scripts/download_data.sh

# train (3 epochs, ~10min)
bash scripts/train_quick.sh

# evaluate
bash scripts/eval_quick.sh

# visualize
python tools/visualize.py --sample-idx 0
```

Train from scratch:
```bash
python tools/train.py --data-root data/nuscenes --work-dir work_dirs/nvautonet --batch-size 2 --epochs 24
```

Evaluate:
```bash
python tools/test.py --data-root data/nuscenes --checkpoint work_dirs/nvautonet/best.pth
```

## structure

```
nvautonet/
├── models/necks/columnwise_mlp.py      # Column-wise MLP (Sec 3.3.3)
├── models/heads/detection_head.py      # Set prediction (Sec 4.1)
├── core/loss/detection_loss.py         # Decomposed losses (Eq 5-7)
├── core/loss/loss_balancer.py          # Adaptive balancing (Eq 14)
├── core/matching/greedy_matcher.py     # Greedy matching (Sec 4.1)
└── utils/bev_utils.py                  # BEV lookup tables
```

## citation

```bibtex
@article{pham2023nvautonet,
  title={NVAutoNet: Fast and Accurate 360° 3D Visual Perception For Self Driving},
  author={Pham, Trung and others},
  journal={arXiv preprint arXiv:2303.12976},
  year={2023}
}
```

"""
nuScenes Dataset Loader
Simplified version for NVAutoNet
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class NuScenesDataset(Dataset):
    """
    nuScenes dataset loader for NVAutoNet.

    This is a simplified version. For full integration, use MMDetection3D's
    NuScenesDataset which handles all the complexities.
    """

    CAMERAS = [
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]

    def __init__(self,
                 data_root,
                 ann_file=None,
                 pipeline=None,
                 split='train',
                 version='v1.0-mini',
                 use_mini=True):
        """
        Args:
            data_root: Root directory of nuScenes data
            ann_file: Annotation file (pickle) from create_data.py
            pipeline: Data augmentation pipeline
            split: 'train' or 'val'
            version: nuScenes version
            use_mini: Whether using mini split
        """
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.version = version

        # Try to import nuScenes devkit
        try:
            from nuscenes.nuscenes import NuScenes
            self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        except ImportError:
            raise ImportError(
                "nuScenes devkit not installed. Run: pip install nuscenes-devkit"
            )

        # Get sample tokens for this split
        if use_mini:
            # For mini split, use first 8 scenes for train, last 2 for val
            all_scenes = self.nusc.scene
            if split == 'train':
                scene_tokens = [s['token'] for s in all_scenes[:8]]
            else:
                scene_tokens = [s['token'] for s in all_scenes[8:]]

            self.samples = []
            for scene_token in scene_tokens:
                scene = self.nusc.get('scene', scene_token)
                sample_token = scene['first_sample_token']

                while sample_token:
                    sample = self.nusc.get('sample', sample_token)
                    self.samples.append(sample)
                    sample_token = sample['next']
        else:
            # For full split, this would require the splits defined in nuscenes
            raise NotImplementedError("Full split not implemented yet")

        # Image transforms
        if pipeline is None:
            self.transform = transforms.Compose([
                transforms.Resize((450, 800)),  # Reduced from (900, 1600) to save GPU memory
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = pipeline

        print(f"[NuScenesDataset] Loaded {len(self.samples)} samples for {split}")

    def __len__(self):
        return len(self.samples)

    def get_camera_data(self, sample_token, camera_name):
        """Get camera image and calibration for a specific camera"""

        sample = self.nusc.get('sample', sample_token)
        cam_token = sample['data'][camera_name]
        cam_data = self.nusc.get('sample_data', cam_token)

        # Image path
        img_path = os.path.join(self.data_root, cam_data['filename'])

        # Camera calibration
        calibrated_sensor_token = cam_data['calibrated_sensor_token']
        calibrated_sensor = self.nusc.get('calibrated_sensor', calibrated_sensor_token)

        # Intrinsic matrix (3x3)
        intrinsic = np.array(calibrated_sensor['camera_intrinsic'], dtype=np.float32)

        # Extrinsic: camera to ego vehicle
        rotation = calibrated_sensor['rotation']  # Quaternion
        translation = calibrated_sensor['translation']

        # Convert quaternion to rotation matrix
        from pyquaternion import Quaternion
        rot_matrix = Quaternion(rotation).rotation_matrix

        # Build 4x4 extrinsic matrix
        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, :3] = rot_matrix
        extrinsic[:3, 3] = translation

        return img_path, intrinsic, extrinsic

    def get_annotations(self, sample_token):
        """Get 3D bounding box annotations"""

        sample = self.nusc.get('sample', sample_token)
        ann_tokens = sample['anns']

        boxes = []
        labels = []

        # DEBUG: Track filtering
        total_anns = len(ann_tokens)
        kept_anns = 0

        for ann_token in ann_tokens:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Category
            category = ann['category_name']

            # Map nuScenes categories to 10 detection classes
            # Reference: https://www.nuscenes.org/nuscenes#data-collection
            if category.startswith('vehicle.car'):
                label = 0
            elif category.startswith('vehicle.truck'):
                label = 1
            elif category.startswith('vehicle.construction'):
                label = 2
            elif category.startswith('vehicle.bus'):
                label = 3
            elif category.startswith('vehicle.trailer'):
                label = 4
            elif category.startswith('vehicle.barrier'):
                label = 5
            elif category.startswith('vehicle.motorcycle') or category.startswith('vehicle.bicycle'):
                label = 6
            elif category.startswith('human.pedestrian'):
                label = 7
            elif category.startswith('movable_object.trafficcone'):
                label = 8
            elif category.startswith('movable_object.barrier'):
                label = 9
            else:
                continue  # Skip categories not in the 10 detection classes

            # 3D box: [x, y, z, w, l, h, yaw]
            center = ann['translation']  # [x, y, z]
            size = ann['size']  # [w, l, h]
            rotation = ann['rotation']  # Quaternion

            # Convert quaternion to yaw angle
            from pyquaternion import Quaternion
            yaw = Quaternion(rotation).yaw_pitch_roll[0]

            box = center + size + [yaw]
            boxes.append(box)
            labels.append(label)
            kept_anns += 1

        if len(boxes) == 0:
            boxes = np.zeros((0, 7), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            if total_anns > 10:  # Only print if there were annotations to filter
                print(f"[Dataset] WARNING: Sample filtered all {total_anns} annotations → 0 boxes")
        else:
            boxes = np.array(boxes, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            if kept_anns < total_anns * 0.2:  # Less than 20% kept
                print(f"[Dataset] Kept only {kept_anns}/{total_anns} annotations after filtering")

        return boxes, labels

    def __getitem__(self, idx):
        """
        Get a training sample.

        Returns:
            data_dict: Dict containing:
                - imgs: Dict[cam_id -> (3, H, W) tensor]
                - camera_ids: List of camera names
                - camera_params: Dict[cam_id -> {'intrinsic', 'extrinsic'}]
                - targets: Dict with 'boxes' and 'labels'
        """
        sample = self.samples[idx]
        sample_token = sample['token']

        # Load multi-camera images
        imgs = {}
        camera_params = {}

        for cam_name in self.CAMERAS:
            img_path, intrinsic, extrinsic = self.get_camera_data(sample_token, cam_name)

            # Load and transform image
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img)

            imgs[cam_name] = img_tensor

            # Store camera parameters
            camera_params[cam_name] = {
                'intrinsic': torch.from_numpy(intrinsic),
                'extrinsic': torch.from_numpy(extrinsic)
            }

        # Get annotations
        boxes, labels = self.get_annotations(sample_token)

        targets = {
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels)
        }

        data_dict = {
            'imgs': imgs,
            'camera_ids': self.CAMERAS,
            'camera_params': camera_params,
            'targets': targets,
            'sample_token': sample_token
        }

        return data_dict


def collate_fn(batch):
    """
    Custom collate function for DataLoader.

    Handles variable number of boxes per sample.
    """
    # Stack images
    imgs = {}
    camera_ids = batch[0]['camera_ids']

    for cam_id in camera_ids:
        imgs[cam_id] = torch.stack([b['imgs'][cam_id] for b in batch])

    # Camera params (same for all samples in batch, just take first)
    camera_params = batch[0]['camera_params']

    # Targets (list of dicts, can't stack due to variable sizes)
    targets = [b['targets'] for b in batch]

    sample_tokens = [b['sample_token'] for b in batch]

    return {
        'imgs': imgs,
        'camera_ids': camera_ids,
        'camera_params': camera_params,
        'targets': targets,
        'sample_tokens': sample_tokens
    }

"""
BEV Utility Functions
Handles coordinate transformations and BEV grid generation
"""

import numpy as np
import torch


def generate_bev_grid(x_bound, y_bound):
    """
    Generate Cartesian BEV grid coordinates.

    Args:
        x_bound: [x_min, x_max, x_resolution] in meters
        y_bound: [y_min, y_max, y_resolution] in meters

    Returns:
        grid_coords: (H, W, 2) array of (x, y) coordinates
        grid_shape: (H, W) BEV grid dimensions
    """
    x_min, x_max, x_res = x_bound
    y_min, y_max, y_res = y_bound

    # Create grid
    xs = np.arange(x_min, x_max, x_res)
    ys = np.arange(y_min, y_max, y_res)

    grid_w = len(xs)  # X dimension
    grid_h = len(ys)  # Y dimension

    # Meshgrid: (H, W) for each coordinate
    grid_x, grid_y = np.meshgrid(xs, ys, indexing='xy')

    # Stack to get (H, W, 2)
    grid_coords = np.stack([grid_x, grid_y], axis=-1)

    return grid_coords, (grid_h, grid_w)


def generate_bev_lookup_cartesian(
    camera_intrinsic,
    camera_extrinsic,
    image_size,
    x_bound,
    y_bound,
    ground_height=0.0
):
    """
    Generate lookup table for pixel-to-BEV mapping (Cartesian grid).

    For nuScenes regular perspective cameras, each image column projects to
    a ray in 3D space. We intersect this ray with the ground plane to get BEV positions.

    Args:
        camera_intrinsic: (3, 3) camera intrinsic matrix K
        camera_extrinsic: (4, 4) camera-to-ego transformation matrix
        image_size: (H, W) image dimensions
        x_bound: [x_min, x_max, x_resolution] in meters
        y_bound: [y_min, y_max, y_resolution] in meters
        ground_height: float, ground plane height (default 0.0)

    Returns:
        lookup_table: (W, H, 2) array mapping (col, row) -> (bev_x_idx, bev_y_idx)
                     Values are -1 if pixel doesn't map to BEV grid
    """
    img_h, img_w = image_size

    # Extract camera parameters
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

    # Extract rotation and translation (camera to ego)
    R = camera_extrinsic[:3, :3]
    t = camera_extrinsic[:3, 3]

    # BEV grid parameters
    x_min, x_max, x_res = x_bound
    y_min, y_max, y_res = y_bound

    # Initialize lookup table (W, H, 2) with -1 (invalid)
    lookup = np.full((img_w, img_h, 2), -1, dtype=np.int32)

    # For each pixel, compute BEV position
    for col in range(img_w):
        for row in range(img_h):
            # Pixel to normalized camera coordinates
            x_cam_norm = (col - cx) / fx
            y_cam_norm = (row - cy) / fy

            # Ray direction in camera frame (points forward)
            # Camera frame: X-right, Y-down, Z-forward
            ray_cam = np.array([x_cam_norm, y_cam_norm, 1.0])
            ray_cam = ray_cam / np.linalg.norm(ray_cam)  # Normalize

            # Transform ray to ego frame
            ray_ego = R @ ray_cam

            # Camera origin in ego frame
            origin_ego = t

            # Intersect ray with ground plane (z = ground_height)
            # Ray equation: P = origin + t * direction
            # Ground plane: z = ground_height
            # Solve: origin_z + t * direction_z = ground_height

            if abs(ray_ego[2]) < 1e-6:  # Ray parallel to ground
                continue

            t_intersect = (ground_height - origin_ego[2]) / ray_ego[2]

            # Check if intersection is in front of camera
            if t_intersect < 0:
                continue

            # Compute intersection point in ego frame
            point_ego = origin_ego + t_intersect * ray_ego

            # Convert to BEV grid indices
            # Ego frame: X-forward, Y-left, Z-up
            # BEV grid: X-axis corresponds to forward, Y-axis to left
            bev_x = point_ego[0]  # Forward
            bev_y = point_ego[1]  # Left

            # Check if within BEV bounds
            if x_min <= bev_x < x_max and y_min <= bev_y < y_max:
                # Convert to grid indices
                bev_x_idx = int((bev_x - x_min) / x_res)
                bev_y_idx = int((bev_y - y_min) / y_res)

                lookup[col, row] = [bev_x_idx, bev_y_idx]

    return lookup


def project_points_to_bev(points_3d, x_bound, y_bound):
    """
    Project 3D points to BEV grid indices.

    Args:
        points_3d: (N, 3) array of 3D points in ego frame
        x_bound: [x_min, x_max, x_resolution]
        y_bound: [y_min, y_max, y_resolution]

    Returns:
        bev_indices: (N, 2) array of (bev_x_idx, bev_y_idx)
        valid_mask: (N,) boolean mask for points within BEV bounds
    """
    x_min, x_max, x_res = x_bound
    y_min, y_max, y_res = y_bound

    # Extract X and Y coordinates
    x_coords = points_3d[:, 0]
    y_coords = points_3d[:, 1]

    # Convert to grid indices
    bev_x_idx = ((x_coords - x_min) / x_res).astype(np.int32)
    bev_y_idx = ((y_coords - y_min) / y_res).astype(np.int32)

    # Create valid mask
    grid_w = int((x_max - x_min) / x_res)
    grid_h = int((y_max - y_min) / y_res)

    valid_mask = (
        (bev_x_idx >= 0) & (bev_x_idx < grid_w) &
        (bev_y_idx >= 0) & (bev_y_idx < grid_h)
    )

    bev_indices = np.stack([bev_x_idx, bev_y_idx], axis=-1)

    return bev_indices, valid_mask


def generate_polar_bev_grid(angular_bins, radial_bins, min_radius, max_radius,
                            radial_spacing='logarithmic'):
    """
    Generate polar BEV grid coordinates (optional implementation).

    This is closer to the paper's approach for long-range detection (200m).
    For 50m nuScenes range, Cartesian is simpler.

    Args:
        angular_bins: int, number of angular bins (e.g., 360 for 1-degree resolution)
        radial_bins: int, number of radial bins (e.g., 64)
        min_radius: float, minimum detection radius in meters
        max_radius: float, maximum detection radius in meters
        radial_spacing: str, 'linear' or 'logarithmic'

    Returns:
        grid_coords: (angular_bins, radial_bins, 2) array of (x, y) coordinates
        angles: (angular_bins,) array of angles in radians
        radii: (radial_bins,) array of radii in meters
    """
    # Angular bins (0 to 2π)
    angles = np.linspace(0, 2 * np.pi, angular_bins, endpoint=False)

    # Radial bins
    if radial_spacing == 'logarithmic':
        # Logarithmic spacing: dense near, sparse far
        radii = np.logspace(
            np.log10(min_radius),
            np.log10(max_radius),
            radial_bins
        )
    else:
        # Linear spacing
        radii = np.linspace(min_radius, max_radius, radial_bins)

    # Create meshgrid: (angular_bins, radial_bins)
    angle_grid, radius_grid = np.meshgrid(angles, radii, indexing='ij')

    # Convert to Cartesian coordinates
    x_grid = radius_grid * np.cos(angle_grid)
    y_grid = radius_grid * np.sin(angle_grid)

    # Stack to get (angular_bins, radial_bins, 2)
    grid_coords = np.stack([x_grid, y_grid], axis=-1)

    return grid_coords, angles, radii


# Torch versions for GPU acceleration

def generate_bev_lookup_cartesian_torch(
    camera_intrinsic,
    camera_extrinsic,
    image_size,
    x_bound,
    y_bound,
    ground_height=0.0,
    device='cuda'
):
    """
    GPU-accelerated version of BEV lookup generation using PyTorch.

    Args:
        camera_intrinsic: (3, 3) tensor
        camera_extrinsic: (4, 4) tensor
        image_size: (H, W)
        x_bound: [x_min, x_max, x_resolution]
        y_bound: [y_min, y_max, y_resolution]
        ground_height: float
        device: str, 'cuda' or 'cpu'

    Returns:
        lookup_table: (W, H, 2) tensor
    """
    img_h, img_w = image_size

    # Extract camera parameters
    fx, fy = camera_intrinsic[0, 0], camera_intrinsic[1, 1]
    cx, cy = camera_intrinsic[0, 2], camera_intrinsic[1, 2]

    R = camera_extrinsic[:3, :3]
    t = camera_extrinsic[:3, 3]

    x_min, x_max, x_res = x_bound
    y_min, y_max, y_res = y_bound

    # Create pixel coordinate grids (H, W) format to match image indexing
    rows = torch.arange(img_h, device=device, dtype=torch.float32)
    cols = torch.arange(img_w, device=device, dtype=torch.float32)
    row_grid, col_grid = torch.meshgrid(rows, cols, indexing='ij')  # Shape: (H, W)

    # Normalize to camera coordinates
    x_cam_norm = (col_grid - cx) / fx
    y_cam_norm = (row_grid - cy) / fy
    ones = torch.ones_like(x_cam_norm)

    # Ray directions in camera frame: (3, H, W)
    ray_cam = torch.stack([x_cam_norm, y_cam_norm, ones], dim=0)
    ray_cam = ray_cam / torch.norm(ray_cam, dim=0, keepdim=True)

    # Transform to ego frame: (3, H, W)
    ray_ego = torch.einsum('ij,jhw->ihw', R, ray_cam)

    # Camera origin in ego frame: (3, 1, 1)
    origin_ego = t.view(3, 1, 1)

    # Intersection with ground plane
    t_intersect = (ground_height - origin_ego[2]) / ray_ego[2]

    # Intersection points in ego frame
    point_ego = origin_ego + t_intersect.unsqueeze(0) * ray_ego

    # Convert to BEV grid indices
    bev_x_idx = ((point_ego[0] - x_min) / x_res).long()
    bev_y_idx = ((point_ego[1] - y_min) / y_res).long()

    # Create valid mask
    grid_w = int((x_max - x_min) / x_res)
    grid_h = int((y_max - y_min) / y_res)

    valid_mask = (
        (t_intersect > 0) &  # In front of camera
        (bev_x_idx >= 0) & (bev_x_idx < grid_w) &
        (bev_y_idx >= 0) & (bev_y_idx < grid_h)
    )

    # Initialize lookup with -1 (invalid) - note: (W, H, 2) for column-wise access
    # But our grids are (H, W), so we need to transpose
    lookup = torch.full((img_w, img_h, 2), -1, dtype=torch.long, device=device)

    # Fill valid entries - transpose from (H, W) to (W, H)
    lookup[:, :, 0] = torch.where(valid_mask, bev_x_idx, torch.tensor(-1, device=device)).T
    lookup[:, :, 1] = torch.where(valid_mask, bev_y_idx, torch.tensor(-1, device=device)).T

    return lookup

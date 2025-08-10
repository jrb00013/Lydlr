#voxel_utils.py
import open3d as o3d
import numpy as np
import torch

def lidar_to_pointcloud(lidar_tensor, sweep_angle=np.pi, points_per_ring=32):
    # Lidar_tensor shape: (B, 1024) distances
    B, N = lidar_tensor.shape
    xyz_batches = []

    for b in range(B):
        distances = lidar_tensor[b].cpu().numpy()
        num_points = len(distances)
        rings = int(np.sqrt(num_points))
        thetas = np.linspace(-sweep_angle/2, sweep_angle/2, num_points)
        phis = np.linspace(-np.pi/6, np.pi/6, num_points)  # Simple vertical scan angle

        # Simple spherical projection with a flat scan
        x = distances * np.cos(phis) * np.cos(thetas)
        y = distances * np.cos(phis) * np.sin(thetas)
        z = distances * np.sin(phis)

        xyz = np.stack([x, y, z], axis=1)
        xyz_batches.append(xyz)

    return torch.from_numpy(np.array(xyz_batches)).float()     # (B, N, 3)

def visualize_voxel_lidar(lidar_tensor, voxel_size=0.2):
    pointcloud = lidar_to_pointcloud(lidar_tensor)[0].numpy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointcloud))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)
    o3d.visualization.draw_geometries([voxel_grid])

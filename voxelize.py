# voxelize.py - Voxelization Module

import numpy as np
import open3d as o3d
from collections import defaultdict, Counter
from pathlib import Path
import json
import time


def voxelize_with_layers(xyz, layers, voxel_size=0.05, strategy='ground_priority'):
    """
    Voxelize point cloud with intelligent layer assignment
    
    Parameters:
        xyz: Point cloud coordinates (N, 3)
        layers: Layer labels (N,) - 0:ground, 1:low_vegetation, 2:mid_vegetation, 3:canopy
        voxel_size: Voxel size in meters (default: 0.05)
        strategy: Layer assignment strategy
            - 'majority_vote': Most common layer in voxel
            - 'ground_priority': Ground maximization strategy (default)
            - 'height_based': Height-based intelligent assignment
    
    Returns:
        voxel_points: Voxel center coordinates (M, 3)
        voxel_layers: Voxel layer labels (M,)
        voxel_info: Voxel statistics dictionary
    """
    start_time = time.time()
    
    # Calculate voxel indices
    voxel_indices = np.floor(xyz / voxel_size).astype(int)
    
    # Create voxel dictionary
    voxel_dict = defaultdict(lambda: {'points': [], 'layers': []})
    
    # Populate voxel dictionary
    for point_idx, (voxel_idx, layer) in enumerate(zip(voxel_indices, layers)):
        voxel_key = tuple(voxel_idx)
        voxel_dict[voxel_key]['points'].append(point_idx)
        voxel_dict[voxel_key]['layers'].append(layer)
    
    # Assign layer labels and compute center points for each voxel
    voxel_points = []
    voxel_layers = []
    layer_stats = Counter()
    voxel_sizes = []
    
    for voxel_key, voxel_data in voxel_dict.items():
        point_indices = voxel_data['points']
        layer_labels = voxel_data['layers']
        
        # Compute voxel center
        voxel_points_coords = xyz[point_indices]
        voxel_center = np.mean(voxel_points_coords, axis=0)
        
        # Assign layer based on strategy
        if strategy == 'majority_vote':
            layer_counts = Counter(layer_labels)
            assigned_layer = layer_counts.most_common(1)[0][0]
        elif strategy == 'ground_priority':
            assigned_layer = _ground_priority_assignment(layer_labels)
        elif strategy == 'height_based':
            assigned_layer = _height_based_assignment(voxel_points_coords, layer_labels, voxel_center)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        voxel_points.append(voxel_center)
        voxel_layers.append(assigned_layer)
        layer_stats[assigned_layer] += 1
        voxel_sizes.append(len(point_indices))
    
    # Convert to numpy arrays
    voxel_points = np.array(voxel_points)
    voxel_layers = np.array(voxel_layers)
    processing_time = time.time() - start_time
    
    # Calculate detailed statistics
    ground_mask_orig = (layers == 0)
    ground_mask_voxel = (voxel_layers == 0)
    
    # Original ground statistics
    if np.any(ground_mask_orig):
        orig_ground = xyz[ground_mask_orig]
        orig_coverage = (orig_ground[:, 0].max() - orig_ground[:, 0].min()) * \
                       (orig_ground[:, 1].max() - orig_ground[:, 1].min())
        orig_density = len(orig_ground) / orig_coverage if orig_coverage > 0 else 0
        ground_thickness = orig_ground[:, 2].max() - orig_ground[:, 2].min()
        avg_point_spacing = np.sqrt(1 / orig_density) if orig_density > 0 else 1.0
    else:
        orig_coverage = 0
        orig_density = 0
        ground_thickness = 0
        avg_point_spacing = 1.0
    
    # Voxelized ground statistics
    if np.any(ground_mask_voxel):
        voxel_ground = voxel_points[ground_mask_voxel]
        voxel_coverage = (voxel_ground[:, 0].max() - voxel_ground[:, 0].min()) * \
                        (voxel_ground[:, 1].max() - voxel_ground[:, 1].min())
        coverage_retention = voxel_coverage / orig_coverage if orig_coverage > 0 else 0
    else:
        voxel_coverage = 0
        coverage_retention = 0
    
    # Generate complete statistics
    voxel_info = {
        'original_points': len(xyz),
        'voxel_count': len(voxel_points),
        'compression_ratio': len(xyz) / len(voxel_points) if len(voxel_points) > 0 else 0,
        'voxel_size': voxel_size,
        'strategy': strategy,
        'processing_time': processing_time,
        'layer_distribution': dict(layer_stats),
        'avg_points_per_voxel': np.mean(voxel_sizes) if voxel_sizes else 0,
        'voxel_size_stats': {
            'min': int(np.min(voxel_sizes)) if voxel_sizes else 0,
            'max': int(np.max(voxel_sizes)) if voxel_sizes else 0,
            'mean': float(np.mean(voxel_sizes)) if voxel_sizes else 0,
            'std': float(np.std(voxel_sizes)) if voxel_sizes else 0
        },
        'ground_analysis': {
            'original_ground_points': int(np.sum(ground_mask_orig)),
            'voxel_ground_points': int(np.sum(ground_mask_voxel)),
            'original_coverage': float(orig_coverage),
            'voxel_coverage': float(voxel_coverage),
            'coverage_retention': float(coverage_retention),
            'ground_thickness': float(ground_thickness),
            'avg_point_spacing': float(avg_point_spacing),
            'original_density': float(orig_density),
            'recommended_voxel_size': float(max(0.01, min(ground_thickness * 5, avg_point_spacing * 3, 1.0)))
        }
    }
    
    return voxel_points, voxel_layers, voxel_info


def _ground_priority_assignment(layer_labels):
    """
    Ground maximization priority strategy
    Prioritize ground if any ground points exist in voxel
    """
    if 0 in layer_labels:
        ground_ratio = np.sum(np.array(layer_labels) == 0) / len(layer_labels)
        if ground_ratio > 0.03:  # Mark as ground if >3% ground points
            return 0
    
    # Otherwise use majority vote
    layer_counts = Counter(layer_labels)
    return layer_counts.most_common(1)[0][0]


def _height_based_assignment(voxel_points_coords, layer_labels, voxel_center):
    """
    Height-based intelligent assignment with ground layer protection
    """
    layer_labels = np.array(layer_labels)
    heights = voxel_points_coords[:, 2]
    
    if 0 in layer_labels:
        ground_mask = (layer_labels == 0)
        ground_ratio = np.sum(ground_mask) / len(layer_labels)
        ground_heights = heights[ground_mask]
        voxel_height_range = heights.max() - heights.min()
        
        ground_in_bottom = np.mean(ground_heights) <= np.percentile(heights, 30)
        thin_voxel = voxel_height_range < 0.15
        significant_ground = ground_ratio > 0.15
        
        if len(ground_heights) > 1:
            ground_layer_thin = (ground_heights.max() - ground_heights.min()) < 0.1
        else:
            ground_layer_thin = True
        
        if (ground_in_bottom and ground_layer_thin) or thin_voxel or significant_ground:
            return 0
    
    layer_counts = Counter(layer_labels)
    return layer_counts.most_common(1)[0][0]


def save_voxelized_data(voxel_points, voxel_layers, voxel_info, output_dir, prefix):
    """
    Save voxelized data to files
    
    Parameters:
        voxel_points: Voxel center coordinates
        voxel_layers: Voxel layer labels
        voxel_info: Voxelization statistics
        output_dir: Output directory path
        prefix: File name prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save voxelized point cloud as PCD
    voxel_pcd_path = output_dir / f"{prefix}_voxelized.pcd"
    voxel_pcd = o3d.geometry.PointCloud()
    voxel_pcd.points = o3d.utility.Vector3dVector(voxel_points)
    
    # Set colors based on layers
    colors = np.zeros((len(voxel_points), 3))
    color_map = {
        0: [0.5, 0.3, 0.1],  # Ground - brown
        1: [0.2, 0.8, 0.2],  # Low vegetation - light green
        2: [0.1, 0.6, 0.1],  # Mid vegetation - medium green
        3: [0.0, 0.4, 0.0]   # Canopy - dark green
    }
    
    for i, layer in enumerate(voxel_layers):
        colors[i] = color_map.get(layer, [0.5, 0.5, 0.5])
    
    voxel_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(voxel_pcd_path), voxel_pcd)
    
    # Save voxel layer information
    voxel_layers_path = output_dir / f"{prefix}_voxelized_layers.npy"
    np.save(voxel_layers_path, voxel_layers)
    
    # Save voxelization statistics as JSON
    info_path = output_dir / f"{prefix}_voxel_info.json"
    with open(info_path, 'w') as f:
        json.dump(voxel_info, f, indent=2)


def recommend_voxel_size_for_ground(points, layers, target_density=100):
    """
    Recommend appropriate voxel size for ground layer preservation
    
    Parameters:
        points: Point cloud coordinates
        layers: Layer labels
        target_density: Target point density per square meter
        
    Returns:
        recommended_size: Recommended voxel size in meters
    """
    ground_mask = (layers == 0)
    if not np.any(ground_mask):
        return 0.05  # Default value
    
    ground_points = points[ground_mask]
    
    # Calculate ground coverage area
    x_range = ground_points[:, 0].max() - ground_points[:, 0].min()
    y_range = ground_points[:, 1].max() - ground_points[:, 1].min()
    area = x_range * y_range
    
    # Calculate voxel size based on target density
    n_ground = len(ground_points)
    current_density = n_ground / area if area > 0 else 100
    
    # Recommend voxel size based on density
    if current_density > 1000:
        return 0.1   # High density - use larger voxels
    elif current_density > 500:
        return 0.05  # Medium density - default
    else:
        return 0.03  # Low density - use smaller voxels


def validate_ground_layer_integrity(original_points, original_layers, 
                                   voxel_points, voxel_layers, voxel_size):
    """
    Validate ground layer preservation after voxelization
    
    Parameters:
        original_points: Original point cloud
        original_layers: Original layer labels
        voxel_points: Voxelized point cloud
        voxel_layers: Voxelized layer labels
        voxel_size: Voxel size used
        
    Returns:
        retention_rate: Ground layer retention rate (0-1)
    """
    original_ground = np.sum(original_layers == 0)
    voxel_ground = np.sum(voxel_layers == 0)
    
    retention_rate = voxel_ground / original_ground if original_ground > 0 else 0
    
    return retention_rate
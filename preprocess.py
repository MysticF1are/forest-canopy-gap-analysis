# preprocess.py - Point Cloud Preprocessing Module

import os
import json
import numpy as np
import laspy
import open3d as o3d
from pathlib import Path

# Optional dependencies
try:
    import pdal
    HAS_PDAL = True
except ImportError:
    HAS_PDAL = False

try:
    import CSF
    HAS_CSF = True
except ImportError:
    HAS_CSF = False

from voxelize import (
    voxelize_with_layers, 
    save_voxelized_data, 
    recommend_voxel_size_for_ground, 
    validate_ground_layer_integrity
)


def detect_ground_csf(xyz, cloth_resolution=0.1, rigidness=3):
    """
    Detect ground points using Cloth Simulation Filter (CSF)
    
    Parameters:
        xyz: Point cloud coordinates (N, 3)
        cloth_resolution: Grid resolution of cloth (default: 0.1)
        rigidness: Rigidness of cloth model (default: 3)
        
    Returns:
        ground_mask: Boolean mask for ground points (N,)
    """
    if not HAS_CSF:
        # Fallback: simple height-based ground detection
        z_threshold = np.percentile(xyz[:, 2], 10)
        return xyz[:, 2] <= z_threshold
    
    csf = CSF.CSF()
    
    # Set CSF parameters
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = cloth_resolution
    csf.params.rigidness = rigidness
    csf.params.time_step = 0.65
    csf.params.class_threshold = 0.03
    csf.params.interations = 500
    
    # Execute CSF
    csf.setPointCloud(xyz)
    ground_idx = CSF.VecInt()
    non_ground_idx = CSF.VecInt()
    csf.do_filtering(ground_idx, non_ground_idx)
    
    # Create boolean mask
    mask = np.zeros(len(xyz), dtype=bool)
    mask[list(ground_idx)] = True
    
    return mask


def laz_to_las(laz_path, las_path):
    """
    Decompress LAZ file to LAS format
    
    Parameters:
        laz_path: Path to input LAZ file
        las_path: Path to output LAS file
        
    Returns:
        success: True if successful, False otherwise
    """
    # Try using laspy first (simpler, pure Python)
    try:
        with laspy.open(laz_path) as laz_file:
            laz_file.read().write(las_path)
        return True
    except:
        pass
    
    # Try PDAL if available
    if HAS_PDAL:
        try:
            pipeline_json = {
                "pipeline": [
                    str(laz_path),
                    {
                        "type": "writers.las",
                        "filename": str(las_path)
                    }
                ]
            }
            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            pipeline.execute()
            return True
        except:
            pass
    
    return False


def classify_vegetation_layers(xyz_centered, ground_mask):
    """
    Classify points into vegetation layers based on height
    
    Parameters:
        xyz_centered: Centered point cloud coordinates (N, 3)
        ground_mask: Boolean mask for ground points
        
    Returns:
        layers: Layer labels (N,) - 0:ground, 1:low, 2:mid, 3:canopy
    """
    z_min = xyz_centered[:, 2].min()
    z_max = xyz_centered[:, 2].max()
    height_range = z_max - z_min
    
    # Initialize all points as canopy (highest layer)
    layers = np.full(len(xyz_centered), 3, dtype=np.int32)
    
    # Classify based on relative height
    # Mid-story: below 75% of height range
    layers[xyz_centered[:, 2] <= z_min + 0.75 * height_range] = 2
    
    # Low vegetation: below 50% of height range
    layers[xyz_centered[:, 2] <= z_min + 0.50 * height_range] = 1
    
    # Ground: identified by CSF
    layers[ground_mask] = 0
    
    return layers


def las_to_pcd_and_ply_with_layers(las_file, output_dir, voxel_size=0.05, progress_callback=None):
    """
    Process LAS file: center, detect ground, classify layers, voxelize
    
    Parameters:
        las_file: Path to input LAS file
        output_dir: Output directory path
        voxel_size: Target voxel size (default: 0.05)
        progress_callback: Optional callback for progress updates
        
    Returns:
        result: Dictionary with output file paths and statistics
    """
    las_file = Path(las_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read LAS file
    if progress_callback:
        progress_callback("Reading LAS file...")
    
    las = laspy.read(las_file)
    xyz = np.vstack((las.x, las.y, las.z)).T
    
    # Center point cloud
    centroid = np.mean(xyz, axis=0)
    xyz_centered = xyz - centroid
    
    # Detect ground points
    if progress_callback:
        progress_callback("Detecting ground points...")
    
    ground_mask = detect_ground_csf(xyz_centered)
    
    # Classify vegetation layers
    if progress_callback:
        progress_callback("Classifying vegetation layers...")
    
    layers = classify_vegetation_layers(xyz_centered, ground_mask)
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_centered)
    
    # Preserve original colors if available
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        colors = np.vstack((las.red, las.green, las.blue)).T / 65535.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default gray color
        colors = np.full((len(xyz_centered), 3), 0.7)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Save original processed data
    if progress_callback:
        progress_callback("Saving processed point cloud...")
    
    pcd_path = output_dir / f"{las_file.stem}_with_layers.pcd"
    ply_path = output_dir / f"{las_file.stem}_with_layers.ply"
    layers_path = output_dir / f"{las_file.stem}_layers.npy"
    
    o3d.io.write_point_cloud(str(pcd_path), pcd)
    o3d.io.write_point_cloud(str(ply_path), pcd)
    np.save(layers_path, layers)
    
    # Voxelization
    if progress_callback:
        progress_callback("Performing voxelization...")
    
    # Get recommended voxel size
    recommended_voxel_size = recommend_voxel_size_for_ground(xyz_centered, layers)
    final_voxel_size = min(recommended_voxel_size, voxel_size, 0.5)
    
    # Perform voxelization
    voxel_points, voxel_layers, voxel_info = voxelize_with_layers(
        xyz_centered, layers,
        voxel_size=final_voxel_size,
        strategy='height_based'
    )
    
    # Validate ground layer integrity
    retention_rate = validate_ground_layer_integrity(
        xyz_centered, layers, voxel_points, voxel_layers, final_voxel_size
    )
    
    # Save voxelized data
    save_voxelized_data(
        voxel_points, voxel_layers, voxel_info,
        output_dir, las_file.stem
    )
    
    if progress_callback:
        progress_callback("Processing complete!")
    
    # Return paths and statistics
    return {
        'pcd': str(pcd_path),
        'ply': str(ply_path),
        'layers': str(layers_path),
        'voxel_pcd': str(output_dir / f"{las_file.stem}_voxelized.pcd"),
        'voxel_layers': str(output_dir / f"{las_file.stem}_voxelized_layers.npy"),
        'statistics': {
            'original_points': len(xyz),
            'ground_points': int(np.sum(ground_mask)),
            'voxel_count': len(voxel_points),
            'voxel_size': final_voxel_size,
            'ground_retention': retention_rate,
            'compression_ratio': voxel_info.get('compression_ratio', 0)
        }
    }


def preprocess_las(input_path, cache_dir=None, progress_callback=None):
    """
    Main preprocessing interface
    
    Parameters:
        input_path: Path to input LAZ/LAS file
        cache_dir: Cache directory (default: ./cache)
        progress_callback: Optional callback function for progress updates
        
    Returns:
        Tuple of (pcd_path, ply_path, layers_path) or
        Dictionary with all processed file paths and statistics
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"File not found: {input_path}")
    
    # Set cache directory
    if cache_dir is None:
        cache_dir = input_path.parent / "cache"
    else:
        cache_dir = Path(cache_dir)
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Handle LAZ decompression if needed
    if input_path.suffix.lower() == '.laz':
        las_path = cache_dir / f"{input_path.stem}.las"
        
        if not las_path.exists():
            if progress_callback:
                progress_callback("Decompressing LAZ file...")
            
            success = laz_to_las(input_path, las_path)
            if not success:
                raise RuntimeError("Failed to decompress LAZ file")
                
    elif input_path.suffix.lower() == '.las':
        las_path = input_path
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")
    
    # Check if already processed
    pcd_path = cache_dir / f"{las_path.stem}_with_layers.pcd"
    ply_path = cache_dir / f"{las_path.stem}_with_layers.ply"
    layers_path = cache_dir / f"{las_path.stem}_layers.npy"
    
    # Process if needed
    if not all(p.exists() for p in [pcd_path, ply_path, layers_path]):
        if progress_callback:
            progress_callback("Processing point cloud...")
        
        result = las_to_pcd_and_ply_with_layers(las_path, cache_dir, progress_callback=progress_callback)
        
        # Return full result dictionary
        return result
    else:
        # Return existing file paths
        return {
            'pcd': str(pcd_path),
            'ply': str(ply_path),
            'layers': str(layers_path),
            'statistics': None  # No statistics for cached files
        }


# For backward compatibility
def laz_to_las(laz_path, las_path):
    """Legacy function for LAZ to LAS conversion"""
    return laz_to_las(laz_path, las_path)


def las_to_pcd_and_ply_with_layers(las_file, output_dir, voxel_size=0.05):
    """Legacy function for processing LAS files"""
    return las_to_pcd_and_ply_with_layers(las_file, output_dir, voxel_size)
# output.py - Data Output Module
import os
import time
import numpy as np
import open3d as o3d
from datetime import datetime
from pathlib import Path

# Import  modules
from solar_ray_system import SolarRaycastingSystem
from optimized_ray_tracing import OptimizedRayTracingSystem
from shadow_map import ShadowMapGenerator
from visualization_maps import display_intensity_distribution, calculate_gap_fraction
from voxelize import voxelize_with_layers

def run_analysis(input_file="cache/segment_with_layers.pcd",
                 output_dir=None,
                 use_solar=False,
                 solar_params=None,
                 ray_spacing=0.05,
                 voxel_size=0.05,
                 use_parallel=True,
                 num_workers=4):
    """
    Run complete light and shadow analysis
    
    Parameters:
        input_file: Input point cloud file path
        output_dir: Output directory (None = auto-generate with timestamp)
        use_solar: Use solar rays (True) or vertical rays (False)
        solar_params: Solar parameters dict (year, month, day, hour, minute, latitude, longitude)
        ray_spacing: Ray spacing in meters
        voxel_size: Voxel size for processing
        use_parallel: Use parallel computation
        num_workers: Number of parallel workers
        
    Returns:
        output_dir: Path to output directory
    """
    
    # Create output directory
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output_{timestamp}")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Start report
    report = []
    report.append("="*60)
    report.append("FOREST CANOPY LIGHT AND SHADOW ANALYSIS")
    report.append("="*60)
    report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Input File: {os.path.basename(input_file)}")
    report.append(f"Output Directory: {output_dir}")
    report.append("")
    
    # Load data
    cache_dir = os.path.dirname(input_file)
    pcd_path = os.path.join(cache_dir, "segment_with_layers.pcd")
    layers_path = os.path.join(cache_dir, "segment_layers.npy")
    
    if os.path.exists(pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        original_points = np.asarray(pcd.points)
    else:
        print(f"Error: {pcd_path} not found")
        return None
    
    if os.path.exists(layers_path):
        original_layers = np.load(layers_path)
    else:
        print(f"Error: {layers_path} not found")
        return None
    
    # Voxelize
    voxel_points, voxel_layers, voxel_info = voxelize_with_layers(
        original_points, original_layers,
        voxel_size=voxel_size,
        strategy='ground_priority'
    )
    
    report.append("DATA SUMMARY")
    report.append("-"*40)
    report.append(f"Original Points: {len(original_points):,}")
    report.append(f"Voxelized Points: {len(voxel_points):,}")
    report.append(f"Compression Ratio: {voxel_info['compression_ratio']:.1f}:1")
    report.append(f"Voxel Size: {voxel_size:.3f} m")
    
    ground_count = voxel_info['layer_distribution'].get(0, 0)
    report.append(f"Ground Voxels: {ground_count:,} ({ground_count/len(voxel_points)*100:.1f}%)")
    report.append("")
    
    # Generate rays
    min_bound = voxel_points.min(axis=0)
    max_bound = voxel_points.max(axis=0)
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    
    if use_solar and solar_params:
        solar_system = SolarRaycastingSystem(
            latitude=solar_params.get('latitude', 0),
            longitude=solar_params.get('longitude', 0)
        )
        ray_origins, ray_direction, ray_info = solar_system.generate_solar_rays(
            aabb,
            year=solar_params.get('year', 2024),
            month=solar_params.get('month', 6),
            day=solar_params.get('day', 21),
            hour=solar_params.get('hour', 12),
            minute=solar_params.get('minute', 0),
            ray_spacing=ray_spacing
        )
        
        report.append("RAY CONFIGURATION")
        report.append("-"*40)
        report.append(f"Type: Solar Rays")
        report.append(f"Solar Elevation: {ray_info['elevation_deg']:.1f}°")
        report.append(f"Solar Azimuth: {ray_info['azimuth_deg']:.1f}°")
        report.append(f"DateTime: {ray_info['datetime']}")
    else:
        solar_system = SolarRaycastingSystem()
        ray_origins, ray_direction, ray_info = solar_system.generate_vertical_test_rays(
            aabb, ray_spacing=ray_spacing
        )
        
        report.append("RAY CONFIGURATION")
        report.append("-"*40)
        report.append("Type: Vertical Test Rays")
    
    report.append(f"Ray Count: {len(ray_origins):,}")
    report.append(f"Ray Spacing: {ray_spacing:.3f} m")
    report.append(f"Ray Direction: [{ray_direction[0]:.3f}, {ray_direction[1]:.3f}, {ray_direction[2]:.3f}]")
    report.append("")
    
    # Calculate light intensity
    tracer = OptimizedRayTracingSystem(voxel_size=voxel_size, debug=False)
    tracer.load_layered_point_cloud(voxel_points, voxel_layers)
    
    start_time = time.time()
    light_distribution = tracer.compute_ground_light_distribution(
        ray_origins, ray_direction,
        batch_size=1000,
        use_parallel=use_parallel,
        num_workers=num_workers
    )
    light_time = time.time() - start_time
    
    report.append("LIGHT INTENSITY RESULTS")
    report.append("-"*40)
    report.append(f"Processing Time: {light_time:.2f} seconds")
    report.append(f"Successful Rays: {len(light_distribution['positions']):,}")
    report.append(f"Success Rate: {light_distribution['metadata']['success_rate']:.1f}%")
    
    if len(light_distribution['positions']) > 0:
        intensities = light_distribution['intensities']
        report.append(f"Mean Intensity: {np.mean(intensities):.3f}")
        report.append(f"Intensity Range: [{np.min(intensities):.3f}, {np.max(intensities):.3f}]")
        report.append(f"Standard Deviation: {np.std(intensities):.3f}")
    report.append("")
    
    # Calculate shadow map
    shadow_generator = ShadowMapGenerator(voxel_size=0.3)
    shadow_generator.load_layered_point_cloud(voxel_points, voxel_layers)
    
    start_time = time.time()
    shadow_distribution = shadow_generator.compute_shadow_map(
        ray_origins, ray_direction,
        batch_size=2000
    )
    shadow_time = time.time() - start_time
    
    report.append("SHADOW MAP RESULTS")
    report.append("-"*40)
    report.append(f"Processing Time: {shadow_time:.2f} seconds")
    report.append(f"Successful Rays: {len(shadow_distribution['positions']):,}")
    report.append(f"Success Rate: {shadow_distribution['metadata']['success_rate']:.1f}%")
    
    if len(shadow_distribution['positions']) > 0:
        shadow_intensities = shadow_distribution['intensities']
        report.append(f"Mean Transmission: {np.mean(shadow_intensities):.3f}")
        report.append(f"Transmission Range: [{np.min(shadow_intensities):.3f}, {np.max(shadow_intensities):.3f}]")
        
        # Calculate gap fraction
        gap_stats = calculate_gap_fraction(shadow_distribution, threshold=0.5)
        if gap_stats:
            report.append("")
            report.append("GAP FRACTION ANALYSIS")
            report.append("-"*40)
            report.append(f"Gap Fraction: {gap_stats['gap_fraction']:.3f} ({gap_stats['gap_fraction']*100:.1f}%)")
            report.append(f"Canopy Cover: {gap_stats['canopy_cover']:.3f} ({gap_stats['canopy_cover']*100:.1f}%)")
            report.append(f"Illuminated Points: {gap_stats['gap_points']:,}")
            report.append(f"Shadowed Points: {gap_stats['shadowed_points']:,}")
            report.append(f"Total Points: {gap_stats['total_points']:,}")
            report.append(f"Threshold Used: {gap_stats['threshold_used']:.2f}")
            
            # Multi-threshold analysis
            for threshold in [0.3, 0.5, 0.7]:
                gap_points = np.sum(shadow_intensities > threshold)
                gap_frac = gap_points / len(shadow_intensities)
                report.append(f"Gap Fraction (>{threshold:.1f}): {gap_frac:.3f} ({gap_frac*100:.1f}%)")
    
    report.append("")
    report.append("="*60)
    report.append("END OF REPORT")
    
    # Save report
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"Report saved to: {report_file}")
    
    # Generate visualizations (save to output directory)
    import matplotlib.pyplot as plt
    current_backend = plt.get_backend()
    plt.switch_backend('Agg')  # Non-interactive backend for saving only
    
    # Save light intensity map
    old_cwd = os.getcwd()
    os.chdir(output_dir)
    
    display_intensity_distribution(
        voxel_points, voxel_layers,
        light_distribution,
        original_points, original_layers
    )
    
    # Save shadow map
    display_intensity_distribution(
        voxel_points, voxel_layers,
        shadow_distribution,
        original_points, original_layers
    )
    
    os.chdir(old_cwd)
    plt.switch_backend(current_backend)  # Restore original backend
    
    print(f"Images saved to: {output_dir}")
    print(f"Analysis complete!")
    
    return str(output_dir)

# Quick test function
def quick_test():
    """Quick test with default parameters"""
    output_dir = run_analysis(
        input_file="cache/segment_with_layers.pcd",
        use_solar=False,
        ray_spacing=0.05,
        voxel_size=0.05,
        use_parallel=True,
        num_workers=4
    )
    return output_dir

if __name__ == "__main__":
    quick_test()
# visualization_maps.py - Simplified Visualization Module

"""
Visualization map generation module
Simple and functional version
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


class VisualizationMaps:
    """
    Simple map generator for intensity visualization
    """
    
    def __init__(self, voxel_size=0.05):
        self.voxel_size = voxel_size
        self.ground_points = None
        self.ground_tree = None
        
    def set_ground_points(self, ground_points):
        """Set ground points for masking"""
        self.ground_points = ground_points
        if ground_points is not None and len(ground_points) > 0:
            self.ground_tree = cKDTree(ground_points[:, :2])
    
    def generate_heatmap(self, distribution, grid_resolution=0.02, 
                        interpolation_method='cubic', smooth_sigma=1.0):
        """Generate heatmap from intensity distribution"""
        positions = distribution['positions']
        intensities = distribution['intensities']
        
        if len(positions) == 0:
            return None
        
        # Create grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        margin = grid_resolution * 2
        x_min, x_max = x_min - margin, x_max + margin
        y_min, y_max = y_min - margin, y_max + margin
        
        x_grid = np.arange(x_min, x_max + grid_resolution, grid_resolution)
        y_grid = np.arange(y_min, y_max + grid_resolution, grid_resolution)
        xx, yy = np.meshgrid(x_grid, y_grid)
        
        # Interpolate
        grid_points = np.column_stack([xx.ravel(), yy.ravel()])
        
        try:
            zi = griddata(
                positions[:, :2], intensities, grid_points,
                method=interpolation_method, fill_value=np.nan
            ).reshape(xx.shape)
        except:
            # Fallback to nearest if cubic fails
            zi = griddata(
                positions[:, :2], intensities, grid_points,
                method='nearest'
            ).reshape(xx.shape)
        
        # Fill NaN values
        if np.any(np.isnan(zi)):
            zi_nearest = griddata(
                positions[:, :2], intensities, grid_points,
                method='nearest'
            ).reshape(xx.shape)
            zi = np.where(np.isnan(zi), zi_nearest, zi)
        
        # Apply smoothing
        if smooth_sigma > 0:
            zi = gaussian_filter(zi, sigma=smooth_sigma)
        
        # Apply ground mask if available
        if self.ground_tree is not None:
            grid_points_2d = np.column_stack([xx.ravel(), yy.ravel()])
            distances, _ = self.ground_tree.query(grid_points_2d)
            mask = distances < self.voxel_size * 1.5
            mask = mask.reshape(xx.shape)
            zi = np.where(mask, zi, np.nan)
        
        return {
            'x_grid': x_grid,
            'y_grid': y_grid,
            'intensity_grid': zi,
            'positions': positions,
            'intensities': intensities
        }
    


def display_intensity_distribution(points, layers, ground_distribution, 
                                  original_points=None, original_layers=None):
    """
    Display intensity distribution (works for both light and shadow)
    """
    # Use original data if available for better projections
    if original_points is not None and original_layers is not None:
        display_points = original_points
        display_layers = original_layers
    else:
        display_points = points
        display_layers = layers
    
    # Extract ground points
    ground_mask = (display_layers == 0)
    ground_points = display_points[ground_mask]
    all_points = display_points
    
    # Create projections
    projected_ground = ground_points.copy()
    projected_ground[:, 2] = 0
    
    projected_all = all_points.copy()
    projected_all[:, 2] = 0
    
    # Create visualization generator
    vis_generator = VisualizationMaps(voxel_size=0.05)
    vis_generator.set_ground_points(ground_points)
    
    # Generate heatmap
    heatmap_data = vis_generator.generate_heatmap(
        ground_distribution,
        grid_resolution=0.02,
        interpolation_method='cubic',
        smooth_sigma=1.0
    )
    
    if heatmap_data is None:
        print("Failed to generate heatmap")
        return
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Set font
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    # Plot 1: Ground projection
    ax1.scatter(projected_ground[:, 0], projected_ground[:, 1], 
               s=0.08, c='darkred', alpha=0.4)
    ax1.set_title(f'Ground Layer (Z=0)\n{len(projected_ground):,} points')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_facecolor('white')
    
    # Plot 2: All points projection
    step = 20 if len(projected_all) > 100000 else 10
    ax2.scatter(projected_all[::step, 0], projected_all[::step, 1], 
               s=0.03, c='darkblue', alpha=0.15)
    ax2.set_title(f'All Points (Z=0)\n{len(projected_all):,} points (showing 1/{step})')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    ax2.set_facecolor('lightgray')
    
    # Plot 3: Intensity heatmap
    extent = [
    heatmap_data['x_grid'].min(), heatmap_data['x_grid'].max(),
    heatmap_data['y_grid'].min(), heatmap_data['y_grid'].max()
    ]

    # Determine visualization type based on metadata
    intensities = heatmap_data['intensities']
    mean_intensity = np.mean(intensities)

    # Check if this is a shadow map
    is_shadow_map = False
    if 'metadata' in ground_distribution:
        method = ground_distribution['metadata'].get('method', '')
        if 'shadow' in method.lower():
            is_shadow_map = True

    # Set colormap and title based on type
    if is_shadow_map:
        cmap = 'gray'
        vmin, vmax = 0.0, 1.0
        title = 'Shadow Distribution Map'
    else:
    # Light intensity always uses hot colormap (orange colors)
        cmap = 'hot'  # Hot colormap: black → red → orange → yellow
        vmin = 0.0    # Start from 0 for full range
        vmax = 1.0    # End at 1
        title = 'Light Intensity Heatmap'

    im3 = ax3.imshow(
        heatmap_data['intensity_grid'],
        extent=extent,
        origin='lower',
        cmap=cmap,
        alpha=0.9,
        aspect='equal',
        vmin=vmin, vmax=vmax
    )
    ax3.set_title(title)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
    cbar3.set_label('Intensity')
    
    # Add statistics
    stats_text = f"Rays: {len(intensities):,}, "
    stats_text += f"Intensity: [{np.min(intensities):.2f}, {np.max(intensities):.2f}], "
    stats_text += f"Mean: {np.mean(intensities):.2f}"
    fig.text(0.5, 0.02, stats_text, ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save with automatic naming
    if mean_intensity > 0.5:
        filename = 'light_intensity_map.png'
    else:
        filename = 'shadow_map.png'
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Image saved: {filename}")
    
    plt.show()

def calculate_gap_fraction(distribution, threshold=0.5):
    """
    Calculate gap fraction from intensity distribution
    
    Parameters:
        distribution: Intensity distribution dictionary
        threshold: Threshold for determining gap (default: 0.5)
        
    Returns:
        Dictionary with gap fraction statistics
    """
    intensities = distribution['intensities']
    
    if len(intensities) == 0:
        return None
    
    # Calculate gap fraction
    gap_points = np.sum(intensities > threshold)
    total_points = len(intensities)
    gap_fraction = gap_points / total_points if total_points > 0 else 0
    
    # Additional statistics
    gap_stats = {
        'gap_fraction': gap_fraction,
        'canopy_cover': 1 - gap_fraction,
        'gap_points': int(gap_points),
        'shadowed_points': int(total_points - gap_points),
        'total_points': int(total_points),
        'threshold_used': threshold,
        'mean_intensity': float(np.mean(intensities)),
        'median_intensity': float(np.median(intensities))
    }
    
    return gap_stats

def show_result_heatmap_only(points, layers, ground_distribution, 
                            original_points=None, original_layers=None):
    """Legacy name for compatibility"""
    return display_intensity_distribution(points, layers, ground_distribution, 
                                         original_points, original_layers)
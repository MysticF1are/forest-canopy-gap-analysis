# shadow_map.py - Shadow Map Generation (Independent Algorithm)

import numpy as np
from scipy.spatial import cKDTree
import time
import math


class ShadowMapGenerator:
    """
    Shadow map generator - Originally a "wrong" ray tracing algorithm
    that produces beautiful shadow effects
    """
    
    def __init__(self, voxel_size=0.3):
        """
        Initialize shadow map generator
        
        Parameters:
            voxel_size: Voxel size for shadow calculations
        """
        self.voxel_size = voxel_size
        self.shadow_voxel_size = voxel_size
        
        # Data storage
        self.points = None
        self.layers = None
        self.ground_points = None
        self.ground_tree = None
        self.all_points_tree = None
        
    def load_layered_point_cloud(self, points, layers):
        """
        Load layered point cloud data
        
        Parameters:
            points: Point cloud coordinates (N, 3)
            layers: Layer labels (N,) - 0:ground, 1:low, 2:mid, 3:canopy
            
        Returns:
            success: Whether loading was successful
        """
        self.points = np.asarray(points)
        self.layers = np.asarray(layers)
        
        # Data validation
        if len(self.points) != len(self.layers):
            return False
        
        # Separate ground layer points
        ground_mask = (self.layers == 0)
        self.ground_points = self.points[ground_mask]
        
        if len(self.ground_points) == 0:
            return False
        
        # Build KD-trees for fast spatial queries
        self.ground_tree = cKDTree(self.ground_points)
        self.all_points_tree = cKDTree(self.points)
        
        return True
    
    def compute_shadow_map(self, ray_origins, ray_direction, batch_size=2000):
        """
        Compute shadow map using ground backtrace method
        
        Parameters:
            ray_origins: Array of ray origins (N, 3)
            ray_direction: Ray direction vector (3,)
            batch_size: Batch processing size
            
        Returns:
            Shadow distribution with same format as ray tracing
        """
        start_time = time.time()
        
        # Get ground layer information
        ground_mask = (self.layers == 0)
        ground_points = self.points[ground_mask]
        
        if len(ground_points) == 0:
            return self._create_empty_result(len(ray_origins), start_time)
        
        ground_z_min = ground_points[:, 2].min()
        ground_z_max = ground_points[:, 2].max()
        ground_z_center = (ground_z_min + ground_z_max) / 2
        
        # Find rays that can reach ground level
        ground_intersections = []
        
        if abs(ray_direction[2]) > 1e-6:
            t_values = (ground_z_center - ray_origins[:, 2]) / ray_direction[2]
            forward_mask = t_values > 0
            
            if np.any(forward_mask):
                valid_origins = ray_origins[forward_mask]
                valid_t = t_values[forward_mask]
                valid_indices = np.where(forward_mask)[0]
                
                hit_points = valid_origins + valid_t[:, None] * ray_direction[None, :]
                
                # Check proximity to actual ground points
                ground_tree = cKDTree(ground_points)
                distances_to_ground, _ = ground_tree.query(hit_points)
                
                ground_threshold = min(2.0, max(0.5, self.shadow_voxel_size * 2))
                near_ground_mask = distances_to_ground < ground_threshold
                
                if np.any(near_ground_mask):
                    final_indices = valid_indices[near_ground_mask]
                    final_origins = valid_origins[near_ground_mask]
                    final_hit_points = hit_points[near_ground_mask]
                    
                    for ray_idx, origin, hit_point in zip(
                        final_indices, final_origins, final_hit_points
                    ):
                        ground_intersections.append((ray_idx, origin, hit_point))
        
        if len(ground_intersections) == 0:
            return self._create_empty_result(len(ray_origins), start_time)
        
        # Calculate shadows
        positions = []
        intensities = []
        
        # Shadow-specific extinction coefficients (the "wrong" parameters that create good shadows)
        extinction_coefficients = {
            0: 0.05,  # Ground layer: minimal attenuation
            1: 1.8,   # Low vegetation: medium attenuation
            2: 1.2,   # Mid vegetation: stronger attenuation
            3: 0.8    # Canopy: strong attenuation
        }
        
        # Process in batches
        for i in range(0, len(ground_intersections), batch_size):
            batch_end = min(i + batch_size, len(ground_intersections))
            batch = ground_intersections[i:batch_end]
            
            for ray_idx, origin, hit_point in batch:
                shadow_intensity = self._calculate_shadow_intensity(
                    origin, hit_point, extinction_coefficients
                )
                
                positions.append(hit_point)
                intensities.append(shadow_intensity)
        
        # Return in standard format
        positions = np.array(positions) if positions else np.array([])
        intensities = np.array(intensities) if intensities else np.array([])
        
        result = {
            'positions': positions,
            'intensities': intensities,
            'metadata': {
                'total_rays': len(ray_origins),
                'successful_rays': len(positions),
                'success_rate': len(positions) / len(ray_origins) * 100 if len(ray_origins) > 0 else 0,
                'processing_time': time.time() - start_time,
                'method': 'shadow_ground_backtrace',
                'voxel_size': self.shadow_voxel_size
            }
        }
        
        return result
    
    def _calculate_shadow_intensity(self, origin, hit_point, extinction_coefficients):
        """
        Calculate shadow intensity along a ray path
        This is the "wrong" algorithm that produces good shadow effects
        
        Parameters:
            origin: Ray origin point
            hit_point: Ground hit point
            extinction_coefficients: Layer-specific extinction coefficients
            
        Returns:
            shadow_intensity: Final shadow intensity value
        """
        # Calculate path vector and distance
        path_vector = hit_point - origin
        total_distance = np.linalg.norm(path_vector)
        
        # Adaptive sampling based on distance
        num_samples = max(10, int(total_distance / self.shadow_voxel_size * 1.5))
        num_samples = min(num_samples, 50)
        
        # Sample points along the path
        sample_alphas = np.linspace(0.1, 0.9, num_samples)
        sample_points = origin[None, :] + sample_alphas[:, None] * path_vector[None, :]
        
        # Query nearest points in the cloud
        distances, nearest_indices = self.all_points_tree.query(sample_points, k=1)
        
        # Determine which samples are within voxels
        valid_mask = distances < self.shadow_voxel_size * 0.6
        
        # Calculate layer distances
        layer_distances = {layer: 0.0 for layer in extinction_coefficients.keys()}
        
        if np.any(valid_mask):
            valid_layers = self.layers[nearest_indices[valid_mask]]
            segment_length = total_distance / num_samples
            
            for layer in valid_layers:
                if layer in layer_distances:
                    layer_distances[layer] += segment_length
        
        # Apply Beer-Lambert law for extinction
        total_extinction = 0.0
        for layer, distance in layer_distances.items():
            if distance > 0 and layer in extinction_coefficients:
                alpha = extinction_coefficients[layer]
                total_extinction += alpha * distance
        
        # Calculate final intensity (exponential decay)
        final_intensity = math.exp(-total_extinction)
        
        return final_intensity
    
    def _create_empty_result(self, total_rays, start_time):
        """
        Create empty result with standard format
        
        Parameters:
            total_rays: Total number of rays attempted
            start_time: Processing start time
            
        Returns:
            Empty result dictionary
        """
        return {
            'positions': np.array([]),
            'intensities': np.array([]),
            'metadata': {
                'total_rays': total_rays,
                'successful_rays': 0,
                'success_rate': 0,
                'processing_time': time.time() - start_time,
                'method': 'shadow_ground_backtrace',
                'voxel_size': self.shadow_voxel_size
            }
        }
    

# optimized_ray_tracing.py - Optimized Ray Tracing System

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp


class OptimizedRayTracingSystem:
    """
    Optimized ray tracing system for light propagation simulation
    
    Core approach:
    1. Rays start from origin and pass through voxels sequentially
    2. Accumulate attenuation in each voxel
    3. Track complete path until reaching ground
    4. Ensure physical accuracy and logical consistency
    
    Performance optimizations:
    - Pre-computed voxel center coordinates
    - Cleaned dual pre-screening logic
    - Reduced redundant calculations
    - Parallel computation support
    """
    
    def __init__(self, voxel_size=0.05, debug=False):
        """
        Initialize ray tracing system
        
        Parameters:
            voxel_size: Voxel size for spatial queries (default: 0.05)
            debug: Enable debug output (default: False)
        """
        self.voxel_size = voxel_size
        self.debug = debug
        
        # Data storage
        self.points = None
        self.layers = None
        self.ground_points = None
        self.ground_tree = None
        self.all_points_tree = None
        
        # Voxel grid for fast ray tracing
        self.voxel_grid = None
        self.voxel_lookup = None
        
        # Pre-computed voxel information
        self.voxel_centers_array = None      # Pre-computed voxel centers (N, 3)
        self.voxel_keys_array = None         # Corresponding voxel keys
        self.voxel_layers_array = None       # Corresponding layer information
        
        # Ground height range for theoretical intersection
        self.ground_z_min = None
        self.ground_z_max = None
        self.ground_z_center = None
    
    def load_layered_point_cloud(self, points, layers):
        """
        Load layered point cloud data and build voxel grid
        
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
        
        # Record ground height range
        self.ground_z_min = self.ground_points[:, 2].min()
        self.ground_z_max = self.ground_points[:, 2].max()
        self.ground_z_center = (self.ground_z_min + self.ground_z_max) / 2
        
        # Build KD-trees for fast spatial queries
        self.ground_tree = cKDTree(self.ground_points)
        self.all_points_tree = cKDTree(self.points)
        
        # Build voxel grid lookup table
        self._build_voxel_grid()
        
        # Pre-compute voxel center arrays
        self._precompute_voxel_arrays()
        
        return True
    
    def _build_voxel_grid(self):
        """Build voxel grid lookup table for fast ray tracing"""
        # Convert each voxel coordinate to grid index
        min_bound = self.points.min(axis=0)
        max_bound = self.points.max(axis=0)
        
        # Calculate grid coordinates for each voxel
        grid_coords = np.floor((self.points - min_bound) / self.voxel_size).astype(int)
        
        # Build lookup dictionary: grid coordinates -> (voxel center, layer)
        self.voxel_lookup = {}
        self.grid_min = min_bound
        
        for i, (point, layer) in enumerate(zip(self.points, self.layers)):
            gx, gy, gz = grid_coords[i]
            grid_key = (gx, gy, gz)
            
            if grid_key not in self.voxel_lookup:
                self.voxel_lookup[grid_key] = {
                    'center': point,
                    'layer': layer,
                    'index': i
                }
        
        self.voxel_grid = list(self.voxel_lookup.keys())
    
    def _precompute_voxel_arrays(self):
        """
        Pre-compute voxel center coordinates and related information
        Avoids repeated calculation for each ray
        """
        # Pre-compute all voxel centers
        self.voxel_centers_array = np.array([
            self.voxel_lookup[key]['center'] for key in self.voxel_grid
        ])
        
        # Pre-compute corresponding keys and layer information
        self.voxel_keys_array = list(self.voxel_grid)
        self.voxel_layers_array = np.array([
            self.voxel_lookup[key]['layer'] for key in self.voxel_grid
        ])
    
    def _calculate_theoretical_ground_point(self, ray_origin, ray_direction):
        """
        Calculate theoretical ground intersection point
        Even if ray is blocked, calculate where it would hit the ground
        
        Parameters:
            ray_origin: Ray starting point
            ray_direction: Ray direction vector
            
        Returns:
            theoretical_point: Theoretical ground intersection (x, y, z)
        """
        # Check if ray is vertical or nearly vertical
        if abs(ray_direction[2]) < 1e-6:
            return None
        
        # Calculate intersection with ground center plane
        t = (self.ground_z_center - ray_origin[2]) / ray_direction[2]
        
        # No intersection if ray goes upward or parallel
        if t <= 0:
            return None
        
        # Calculate theoretical intersection
        theoretical_point = ray_origin + t * ray_direction
        
        return theoretical_point
    
    def trace_ray_through_all_voxels(self, ray_origin, ray_direction, extinction_coefficients=None):
        """
        Trace single ray through all voxels until reaching ground
        Using Beer-Lambert law: I = I₀ * exp(-α * d)
        
        Parameters:
            ray_origin: Ray starting point
            ray_direction: Ray direction vector
            extinction_coefficients: Layer extinction coefficients (1/m)
            
        Returns:
            Result dictionary with ground hit point and intensity
        """
        if extinction_coefficients is None:
            # Now using actual extinction coefficients (1/m) instead of transmittance
            extinction_coefficients = {
                1: 7.13,   # Low vegetation: -ln(0.70)/0.05 ≈ 7.13 /m
                2: 5.75,   # Mid vegetation: -ln(0.75)/0.05 ≈ 5.75 /m
                3: 4.46    # Canopy: -ln(0.80)/0.05 ≈ 4.46 /m
            }
        
        intersections = self._find_ray_voxel_intersections(ray_origin, ray_direction)
        
        # Calculate theoretical ground point
        theoretical_ground_point = self._calculate_theoretical_ground_point(ray_origin, ray_direction)
        
        if theoretical_ground_point is None:
            return None
        
        # If no voxel intersections, ray reaches ground directly
        if len(intersections) == 0:
            dist, _ = self.ground_tree.query(theoretical_ground_point, k=1)
            if dist < self.voxel_size * 2:
                return {
                    'ground_hit_point': theoretical_ground_point,
                    'final_intensity': 1.0,  # No attenuation
                    'is_blocked': False,
                    'path_info': {
                        'layers_passed': [],
                        'total_layers': 0,
                        'segments': [],
                        'total_optical_depth': 0.0
                    }
                }
            else:
                return None
        
        # Sort intersections by distance
        intersections.sort(key=lambda x: x['distance'])
        
        # Beer-Lambert law: accumulate optical depth (α * d)
        total_optical_depth = 0.0
        layers_passed = set()
        ground_hit = None
        path_segments = []
        
        # Track previous position for distance calculation
        previous_position = ray_origin
        
        # Process each intersection
        for intersection in intersections:
            voxel_layer = intersection['layer']
            hit_point = intersection['hit_point']
            voxel_center = intersection['center']
            
            # Calculate actual distance traveled in this voxel
            # Use entry and exit points for accurate distance
            t_near, t_far = self._ray_aabb_optimized(ray_origin, ray_direction, voxel_center)
            
            # Calculate entry and exit points
            if t_near >= 0:
                entry_point = ray_origin + t_near * ray_direction
                exit_point = ray_origin + t_far * ray_direction
                distance_in_voxel = np.linalg.norm(exit_point - entry_point)
            else:
                # Ray origin is inside the voxel
                exit_point = ray_origin + t_far * ray_direction
                distance_in_voxel = np.linalg.norm(exit_point - ray_origin)
            
            # Apply Beer-Lambert attenuation for non-ground layers
            if voxel_layer != 0:
                if voxel_layer in extinction_coefficients:
                    alpha = extinction_coefficients[voxel_layer]
                    optical_depth_contribution = alpha * distance_in_voxel
                    total_optical_depth += optical_depth_contribution
                    
                    if voxel_layer not in layers_passed:
                        layers_passed.add(voxel_layer)
                    
                    # Calculate intensity after this segment
                    current_intensity = np.exp(-total_optical_depth)
                    
                    path_segments.append({
                        'layer': voxel_layer,
                        'distance': distance_in_voxel,
                        'alpha': alpha,
                        'optical_depth': optical_depth_contribution,
                        'cumulative_optical_depth': total_optical_depth,
                        'intensity_after': current_intensity
                    })
            
            # Check if reached ground
            if voxel_layer == 0:
                ground_hit = hit_point
                break
        
        # Calculate final intensity using Beer-Lambert law
        final_intensity = np.exp(-total_optical_depth)
        
        # Handle case where ray doesn't reach ground
        if ground_hit is None:
            dist, _ = self.ground_tree.query(theoretical_ground_point, k=1)
            
            if dist < self.voxel_size * 3:
                # Ray is blocked but would hit ground
                return {
                    'ground_hit_point': theoretical_ground_point,
                    'final_intensity': final_intensity,
                    'is_blocked': True,
                    'path_info': {
                        'layers_passed': list(layers_passed),
                        'total_layers': len(layers_passed),
                        'segments': path_segments,
                        'total_optical_depth': total_optical_depth
                    }
                }
            else:
                return None
        
        # Normal case - ray reaches ground
        return {
            'ground_hit_point': ground_hit,
            'final_intensity': final_intensity,
            'is_blocked': False,
            'path_info': {
                'layers_passed': list(layers_passed),
                'total_layers': len(layers_passed),
                'segments': path_segments,
                'total_optical_depth': total_optical_depth
            }
        }
    
    def _find_ray_voxel_intersections(self, ray_origin, ray_direction):
        """
        Find all voxels intersected by ray
        Optimized with pre-computed arrays and efficient screening
        
        Parameters:
            ray_origin: Ray starting point
            ray_direction: Ray direction vector
            
        Returns:
            List of intersection dictionaries
        """
        # Use pre-computed voxel arrays
        all_voxel_centers = self.voxel_centers_array
        all_voxel_keys = self.voxel_keys_array
        all_voxel_layers = self.voxel_layers_array
        
        # Ray bounding box pre-screening
        ray_length = 50.0  # Estimated maximum ray length
        ray_end = ray_origin + ray_length * ray_direction
        
        ray_min = np.minimum(ray_origin, ray_end) - self.voxel_size * 2
        ray_max = np.maximum(ray_origin, ray_end) + self.voxel_size * 2
        
        # Vectorized bounding box filtering
        bbox_mask = (
            (all_voxel_centers[:, 0] >= ray_min[0]) & (all_voxel_centers[:, 0] <= ray_max[0]) &
            (all_voxel_centers[:, 1] >= ray_min[1]) & (all_voxel_centers[:, 1] <= ray_max[1]) &
            (all_voxel_centers[:, 2] >= ray_min[2]) & (all_voxel_centers[:, 2] <= ray_max[2])
        )
        
        if not np.any(bbox_mask):
            return []
        
        # Apply bounding box filter
        bbox_centers = all_voxel_centers[bbox_mask]
        bbox_keys = [all_voxel_keys[i] for i in np.where(bbox_mask)[0]]
        bbox_layers = all_voxel_layers[bbox_mask]
        
        # Distance pre-screening
        distances = self._dist_to_ray_vectorized(bbox_centers, ray_origin, ray_direction)
        
        # Filter by distance threshold
        dist_limit = self.voxel_size * np.sqrt(3) * 0.5  # Half of voxel diagonal
        candidate_mask = distances <= dist_limit
        
        if not np.any(candidate_mask):
            return []
        
        # Perform precise AABB intersection tests
        candidate_centers = bbox_centers[candidate_mask]
        candidate_keys = [bbox_keys[i] for i in np.where(candidate_mask)[0]]
        candidate_layers = bbox_layers[candidate_mask]
        
        intersections = []
        
        for voxel_center, key, voxel_layer in zip(candidate_centers, candidate_keys, candidate_layers):
            # Optimized AABB test
            t_near, t_far = self._ray_aabb_optimized(ray_origin, ray_direction, voxel_center)
            
            # Check if intersection exists and is in front of ray
            if t_near < t_far and t_far >= 0:
                t_intersection = t_near if t_near >= 0 else t_far
                hit_point = ray_origin + t_intersection * ray_direction
                
                intersections.append({
                    'center': voxel_center,
                    'layer': voxel_layer,
                    'hit_point': hit_point,
                    'distance': t_intersection
                })
        
        return intersections
    
    def _dist_to_ray_vectorized(self, points, ray_origin, ray_direction):
        """
        Vectorized calculation of point distances to ray
        
        Parameters:
            points: Point array (N, 3)
            ray_origin: Ray origin (3,)
            ray_direction: Ray direction (3,)
        
        Returns:
            distances: Distance array (N,)
        """
        points = np.asarray(points)
        ray_origin = np.asarray(ray_origin).flatten()
        ray_direction = np.asarray(ray_direction).flatten()
        
        # Normalize direction vector
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Vectors from ray origin to points
        point_vectors = points - ray_origin[np.newaxis, :]
        
        # Calculate projection lengths (dot product)
        projections = np.dot(point_vectors, ray_direction)
        
        # Calculate projection points
        projection_points = ray_origin + projections[:, np.newaxis] * ray_direction
        
        # Calculate perpendicular distances
        distances = np.linalg.norm(points - projection_points, axis=1)
        
        return distances
    
    def _ray_aabb_optimized(self, ray_origin, ray_direction, voxel_center):
        """
        Optimized ray-AABB intersection test
        
        Parameters:
            ray_origin: Ray origin
            ray_direction: Ray direction
            voxel_center: Voxel center point
            
        Returns:
            t_near, t_far: Entry and exit parameters
        """
        # Calculate voxel bounds with tolerance
        tolerance = self.voxel_size * 0.05  # 5% tolerance
        half_size = self.voxel_size / 2 + tolerance
        
        box_min = voxel_center - half_size
        box_max = voxel_center + half_size
        
        # Avoid division by zero
        ray_dir_safe = np.where(np.abs(ray_direction) < 1e-10, 1e-10, ray_direction)
        
        # Calculate t values
        t_min = (box_min - ray_origin) / ray_dir_safe
        t_max = (box_max - ray_origin) / ray_dir_safe
        
        # Ensure t_min < t_max
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)
        
        # Calculate overall entry and exit times
        t_near = np.max(t1)
        t_far = np.min(t2)
        
        return t_near, t_far
    
    def compute_ground_light_distribution(self, ray_origins, ray_direction, 
                                        extinction_coefficients=None, batch_size=1000,
                                        use_parallel=True, num_workers=None):
        """
        Main function: Compute ground light distribution
        
        Parameters:
            ray_origins: Array of ray origins
            ray_direction: Ray direction vector
            extinction_coefficients: Layer extinction coefficients
            batch_size: Batch size for serial processing
            use_parallel: Enable parallel processing
            num_workers: Number of worker threads
            
        Returns:
            Dictionary with positions, intensities, and statistics
        """
        start_time = time.time()
        
        # Choose execution method
        if use_parallel and len(ray_origins) > 100:
            result = self._compute_parallel(
                ray_origins, ray_direction, 
                extinction_coefficients, num_workers
            )
        else:
            result = self._compute_serial(
                ray_origins, ray_direction, 
                extinction_coefficients, batch_size
            )
        
        # Calculate detailed statistics
        statistics = self._calculate_ray_statistics(result)
        
        # Add metadata
        result['metadata'] = {
            'total_rays': len(ray_origins),
            'successful_rays': len(result['positions']),
            'success_rate': len(result['positions']) / len(ray_origins) * 100 if len(ray_origins) > 0 else 0,
            'processing_time': time.time() - start_time,
            'extinction_coefficients': extinction_coefficients,
            'voxel_size': self.voxel_size,
            'method': 'parallel' if use_parallel else 'serial'
        }
        
        result['statistics'] = statistics
        
        return result
    
    def _calculate_ray_statistics(self, result):
        """
        Calculate ray tracing statistics
        
        Parameters:
            result: Ray tracing result dictionary
            
        Returns:
            statistics: Detailed statistics dictionary
        """
        intensities = result['intensities']
        path_infos = result['path_infos']
        
        if len(intensities) == 0:
            return {}
        
        total_rays = len(intensities)
        
        # Attenuation analysis
        no_attenuation_mask = (intensities >= 0.999)
        no_attenuation_count = np.sum(no_attenuation_mask)
        no_attenuation_ratio = no_attenuation_count / total_rays * 100
        
        attenuation_rates = 1.0 - intensities
        mean_attenuation = np.mean(attenuation_rates) * 100
        
        # Layer penetration analysis
        passed_canopy = 0
        passed_all_layers = 0
        layer_combinations = {}
        
        for path_info in path_infos:
            layers = sorted(path_info.get('layers_passed', []))
            
            if 3 in layers:
                passed_canopy += 1
            
            if set(layers) == {1, 2, 3}:
                passed_all_layers += 1
            
            layer_key = tuple(layers)
            layer_combinations[layer_key] = layer_combinations.get(layer_key, 0) + 1
        
        # Intensity distribution
        intensity_distribution = {
            'no_attenuation': np.sum(intensities > 0.99),
            'light_attenuation': np.sum((intensities > 0.8) & (intensities <= 0.99)),
            'medium_attenuation': np.sum((intensities > 0.6) & (intensities <= 0.8)),
            'strong_attenuation': np.sum((intensities > 0.4) & (intensities <= 0.6)),
            'very_strong_attenuation': np.sum(intensities <= 0.4)
        }
        
        # Canopy gap analysis
        canopy_gap_count = np.sum((intensities > 0.95) & 
                                  [3 not in info.get('layers_passed', []) for info in path_infos])
        
        # Blocking statistics
        blocked_count = np.sum(result.get('is_blocked', np.zeros(len(intensities), dtype=bool)))
        blocked_ratio = blocked_count / total_rays * 100 if total_rays > 0 else 0
        
        statistics = {
            'total_rays': total_rays,
            'mean_intensity': float(np.mean(intensities)),
            'std_intensity': float(np.std(intensities)),
            'min_intensity': float(np.min(intensities)),
            'max_intensity': float(np.max(intensities)),
            'mean_attenuation_rate': float(mean_attenuation),
            'no_attenuation_count': int(no_attenuation_count),
            'no_attenuation_ratio': float(no_attenuation_ratio),
            'passed_canopy_count': int(passed_canopy),
            'passed_canopy_ratio': float(passed_canopy / total_rays * 100),
            'passed_all_layers_count': int(passed_all_layers),
            'passed_all_layers_ratio': float(passed_all_layers / total_rays * 100),
            'intensity_distribution': intensity_distribution,
            'layer_combinations': dict(sorted(layer_combinations.items(), 
                                            key=lambda x: x[1], 
                                            reverse=True)[:10]),
            'canopy_gap_count': int(canopy_gap_count),
            'canopy_gap_ratio': float(canopy_gap_count / total_rays * 100),
            'blocked_count': int(blocked_count),
            'blocked_ratio': float(blocked_ratio)
        }
        
        return statistics
    
    def _compute_serial(self, ray_origins, ray_direction, extinction_coefficients, batch_size):
        """
        Serial computation version
        
        Parameters:
            ray_origins: Ray origins array
            ray_direction: Ray direction
            extinction_coefficients: Extinction coefficients
            batch_size: Batch size
            
        Returns:
            Result dictionary
        """
        positions = []
        intensities = []
        ray_indices = []
        path_infos = []
        is_blocked = []
        
        total_batches = (len(ray_origins) + batch_size - 1) // batch_size
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(ray_origins))
            batch_origins = ray_origins[start_idx:end_idx]
            
            for local_idx, ray_origin in enumerate(batch_origins):
                ray_idx = start_idx + local_idx
                
                result = self.trace_ray_through_all_voxels(
                    ray_origin, ray_direction, extinction_coefficients
                )
                
                if result is not None:
                    positions.append(result['ground_hit_point'])
                    intensities.append(result['final_intensity'])
                    ray_indices.append(ray_idx)
                    path_infos.append(result['path_info'])
                    is_blocked.append(result.get('is_blocked', False))
        
        return {
            'positions': np.array(positions) if positions else np.array([]),
            'intensities': np.array(intensities) if intensities else np.array([]),
            'ray_indices': np.array(ray_indices) if ray_indices else np.array([]),
            'path_infos': path_infos,
            'is_blocked': np.array(is_blocked) if is_blocked else np.array([])
        }
    
    def _compute_parallel(self, ray_origins, ray_direction, extinction_coefficients, num_workers):
        """
        Parallel computation version
        
        Parameters:
            ray_origins: Ray origins array
            ray_direction: Ray direction
            extinction_coefficients: Extinction coefficients
            num_workers: Number of worker threads
            
        Returns:
            Result dictionary
        """
        # Auto-determine worker count
        if num_workers is None:
            num_workers = min(mp.cpu_count(), 8)
        
        # Calculate chunk size
        n_rays = len(ray_origins)
        chunk_size = max(50, n_rays // (num_workers * 4))
        
        # Create task list
        tasks = []
        for i in range(0, n_rays, chunk_size):
            end_idx = min(i + chunk_size, n_rays)
            tasks.append((i, end_idx, ray_origins[i:end_idx]))
        
        # Parallel execution
        all_results = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(
                    self._process_ray_chunk,
                    chunk[2], ray_direction,
                    extinction_coefficients, chunk[0]
                ): chunk for chunk in tasks
            }
            
            for future in as_completed(futures):
                chunk_result = future.result()
                all_results.append(chunk_result)
        
        return self._merge_parallel_results(all_results)
    
    def _process_ray_chunk(self, ray_origins_chunk, ray_direction, extinction_coefficients, start_idx):
        """
        Process a chunk of rays for parallel computation
        
        Parameters:
            ray_origins_chunk: Chunk of ray origins
            ray_direction: Ray direction
            extinction_coefficients: Extinction coefficients
            start_idx: Starting index in original array
            
        Returns:
            Chunk result dictionary
        """
        positions = []
        intensities = []
        ray_indices = []
        path_infos = []
        is_blocked = []
        
        for local_idx, ray_origin in enumerate(ray_origins_chunk):
            result = self.trace_ray_through_all_voxels(
                ray_origin, ray_direction, extinction_coefficients
            )
            
            if result is not None:
                positions.append(result['ground_hit_point'])
                intensities.append(result['final_intensity'])
                ray_indices.append(start_idx + local_idx)
                path_infos.append(result['path_info'])
                is_blocked.append(result.get('is_blocked', False))
        
        return {
            'positions': positions,
            'intensities': intensities,
            'ray_indices': ray_indices,
            'path_infos': path_infos,
            'is_blocked': is_blocked
        }
    
    def _merge_parallel_results(self, results_list):
        """
        Merge results from parallel computation
        
        Parameters:
            results_list: List of chunk results
            
        Returns:
            Merged result dictionary
        """
        results_list.sort(key=lambda x: x['ray_indices'][0] if len(x['ray_indices']) > 0 else float('inf'))
        
        all_positions = []
        all_intensities = []
        all_ray_indices = []
        all_path_infos = []
        all_is_blocked = []
        
        for result in results_list:
            if len(result['positions']) > 0:
                all_positions.extend(result['positions'])
                all_intensities.extend(result['intensities'])
                all_ray_indices.extend(result['ray_indices'])
                all_path_infos.extend(result['path_infos'])
                all_is_blocked.extend(result.get('is_blocked', [False] * len(result['positions'])))
        
        return {
            'positions': np.array(all_positions) if all_positions else np.array([]),
            'intensities': np.array(all_intensities) if all_intensities else np.array([]),
            'ray_indices': np.array(all_ray_indices) if all_ray_indices else np.array([]),
            'path_infos': all_path_infos,
            'is_blocked': np.array(all_is_blocked) if all_is_blocked else np.array([])
        }
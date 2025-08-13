# solar_ray_system.py - Solar Ray Generation System

import numpy as np
import math
import time


class SolarRaycastingSystem:
    """
    Solar ray generation system for light simulation
    
    Features:
    1. Real solar position calculation
    2. Vertical test ray generation
    3. Custom angle ray generation
    4. Flexible parameter configuration
    """
    
    def __init__(self, latitude=51.5074, longitude=-0.1278, timezone_offset=0):
        """
        Initialize solar ray system
        
        Parameters:
            latitude: Latitude in degrees (default: 51.5074 - London)
            longitude: Longitude in degrees (default: -0.1278 - London)
            timezone_offset: Hours offset from UTC (default: 0)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone_offset = timezone_offset
    
    def calculate_solar_position(self, year=2024, month=6, day=21, hour=14, minute=0):
        """
        Calculate solar elevation and azimuth angles
        
        Parameters:
            year: Year (default: 2024)
            month: Month 1-12 (default: 6 - June)
            day: Day of month (default: 21 - summer solstice)
            hour: Hour 0-23 (default: 14)
            minute: Minute 0-59 (default: 0)
            
        Returns:
            solar_elevation: Solar elevation angle in radians
            solar_azimuth: Solar azimuth angle in radians
        """
        # Julian day calculation
        if month <= 2:
            year -= 1
            month += 12
        
        A = int(year / 100)
        B = 2 - A + int(A / 4)
        
        JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
        JD += (hour + minute/60 - self.timezone_offset) / 24
        
        # Solar position calculation
        n = JD - 2451545.0
        L = (280.460 + 0.9856474 * n) % 360
        g = math.radians((357.528 + 0.9856003 * n) % 360)
        
        lambda_sun = math.radians(L + 1.915 * math.sin(g) + 0.020 * math.sin(2*g))
        delta = math.asin(math.sin(lambda_sun) * math.sin(math.radians(23.439)))
        
        solar_time = (hour + minute/60) + (self.longitude - 15 * self.timezone_offset) / 15
        hour_angle = math.radians(15 * (solar_time - 12))
        
        lat_rad = math.radians(self.latitude)
        solar_elevation = math.asin(
            math.sin(lat_rad) * math.sin(delta) + 
            math.cos(lat_rad) * math.cos(delta) * math.cos(hour_angle)
        )
        
        solar_azimuth = math.atan2(
            math.sin(hour_angle),
            math.cos(hour_angle) * math.sin(lat_rad) - math.tan(delta) * math.cos(lat_rad)
        )
        
        return solar_elevation, solar_azimuth
    
    def generate_vertical_test_rays(self, aabb, ray_spacing=0.025, margin=1.0, 
                                  emission_height_offset=10.0, debug=False):
        """
        Generate vertical test rays for algorithm validation
        
        Parameters:
            aabb: Axis-aligned bounding box of target area
            ray_spacing: Distance between rays in meters (default: 0.025)
            margin: Boundary extension in meters (default: 1.0)
            emission_height_offset: Height offset above max Z (default: 10.0)
            debug: Enable debug output (default: False)
            
        Returns:
            ray_origins: Array of ray origins (N, 3)
            ray_direction: Ray direction vector (3,) - downward [0, 0, -1]
            ray_info: Dictionary with ray generation information
        """
        # Get bounding box information
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        extent = max_bound - min_bound
        center = (min_bound + max_bound) / 2
        
        # Calculate emission area
        x_min = min_bound[0] - margin
        x_max = max_bound[0] + margin
        y_min = min_bound[1] - margin
        y_max = max_bound[1] + margin
        z_emission = max_bound[2] + emission_height_offset
        
        # Generate ray grid
        x_coords = np.arange(x_min, x_max + ray_spacing, ray_spacing)
        y_coords = np.arange(y_min, y_max + ray_spacing, ray_spacing)
        
        ray_origins = []
        for x in x_coords:
            for y in y_coords:
                ray_origins.append([x, y, z_emission])
        
        ray_origins = np.array(ray_origins)
        ray_direction = np.array([0.0, 0.0, -1.0])  # Vertical downward
        
        # Compile ray information
        ray_info = {
            'type': 'vertical_test',
            'ray_count': len(ray_origins),
            'ray_spacing': ray_spacing,
            'margin': margin,
            'emission_height': z_emission,
            'coverage_area': {
                'x_range': [x_min, x_max],
                'y_range': [y_min, y_max],
                'total_area': (x_max - x_min) * (y_max - y_min)
            },
            'grid_size': {
                'x_steps': len(x_coords),
                'y_steps': len(y_coords)
            },
            'target_bounds': {
                'min': min_bound.tolist(),
                'max': max_bound.tolist(),
                'center': center.tolist(),
                'extent': extent.tolist()
            }
        }
        
        return ray_origins, ray_direction, ray_info
    
    def generate_custom_angle_rays(self, aabb, elevation_deg, azimuth_deg, 
                                 ray_spacing=0.025, coverage_multiplier=3.0, 
                                 emission_height_offset=10.0, debug=False):
        """
        Generate rays at custom angles
        
        Parameters:
            aabb: Axis-aligned bounding box of target area
            elevation_deg: Elevation angle in degrees (0-90)
            azimuth_deg: Azimuth angle in degrees (0-360, 0=North, 90=East)
            ray_spacing: Distance between rays in meters (default: 0.025)
            coverage_multiplier: Coverage area multiplier (default: 3.0)
            emission_height_offset: Height offset above max Z (default: 10.0)
            debug: Enable debug output (default: False)
            
        Returns:
            ray_origins: Array of ray origins (N, 3)
            ray_direction: Ray direction vector (3,)
            ray_info: Dictionary with ray generation information
        """
        # Convert angles to radians
        elev_rad = math.radians(elevation_deg)
        azim_rad = math.radians(azimuth_deg)
        
        # Calculate light source direction (from ground to source)
        light_direction = np.array([
            math.sin(azim_rad) * math.cos(elev_rad),
            math.cos(azim_rad) * math.cos(elev_rad),
            math.sin(elev_rad)
        ])
        light_direction = light_direction / np.linalg.norm(light_direction)
        
        # Ray direction (from source to ground)
        ray_direction = -light_direction
        
        # Generate ray origins
        ray_origins = self._generate_ray_origins_for_angle(
            aabb, ray_direction, ray_spacing, coverage_multiplier, 
            emission_height_offset, debug
        )
        
        # Compile ray information
        ray_info = {
            'type': 'custom_angle',
            'elevation_deg': elevation_deg,
            'azimuth_deg': azimuth_deg,
            'elevation_rad': elev_rad,
            'azimuth_rad': azim_rad,
            'light_direction': light_direction.tolist(),
            'ray_direction': ray_direction.tolist(),
            'ray_count': len(ray_origins),
            'ray_spacing': ray_spacing,
            'coverage_multiplier': coverage_multiplier,
            'emission_height_offset': emission_height_offset
        }
        
        return ray_origins, ray_direction, ray_info
    
    def generate_solar_rays(self, aabb, year=2024, month=6, day=21, hour=12, minute=0,
                           ray_spacing=0.025, coverage_multiplier=None, 
                           emission_height_offset=10.0, debug=False):
        """
        Generate rays based on real solar position
        
        Parameters:
            aabb: Axis-aligned bounding box of target area
            year: Year (default: 2024)
            month: Month 1-12 (default: 6)
            day: Day of month (default: 21)
            hour: Hour 0-23 (default: 12)
            minute: Minute 0-59 (default: 0)
            ray_spacing: Distance between rays in meters (default: 0.025)
            coverage_multiplier: Coverage area multiplier (None=auto)
            emission_height_offset: Height offset above max Z (default: 10.0)
            debug: Enable debug output (default: False)
            
        Returns:
            ray_origins: Array of ray origins (N, 3)
            ray_direction: Ray direction vector (3,)
            solar_info: Dictionary with solar and ray information
        """
        # Calculate solar position
        solar_elevation, solar_azimuth = self.calculate_solar_position(
            year, month, day, hour, minute
        )
        
        # Calculate sun direction (from ground to sun)
        sun_direction = np.array([
            math.sin(solar_azimuth) * math.cos(solar_elevation),
            math.cos(solar_azimuth) * math.cos(solar_elevation),
            math.sin(solar_elevation)
        ])
        sun_direction = sun_direction / np.linalg.norm(sun_direction)
        
        # Ray direction (from sun to ground)
        ray_direction = -sun_direction
        
        # Auto-determine coverage multiplier based on sun angle
        if coverage_multiplier is None:
            angle_factor = abs(ray_direction[2])  # Z component (verticality)
            if angle_factor < 0.5:  # Low angle sun
                coverage_multiplier = 5.0
            elif angle_factor < 0.7:  # Medium angle
                coverage_multiplier = 4.0
            else:  # High angle sun
                coverage_multiplier = 3.0
        
        # Generate ray origins
        ray_origins = self._generate_ray_origins_for_angle(
            aabb, ray_direction, ray_spacing, coverage_multiplier, 
            emission_height_offset, debug
        )
        
        # Compile solar information
        solar_info = {
            'type': 'solar',
            'elevation_deg': math.degrees(solar_elevation),
            'azimuth_deg': math.degrees(solar_azimuth),
            'elevation_rad': solar_elevation,
            'azimuth_rad': solar_azimuth,
            'sun_direction': sun_direction.tolist(),
            'ray_direction': ray_direction.tolist(),
            'datetime': f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}",
            'location': f"{self.latitude:.4f}N, {abs(self.longitude):.4f}{'W' if self.longitude < 0 else 'E'}",
            'ray_count': len(ray_origins),
            'ray_spacing': ray_spacing,
            'coverage_multiplier': coverage_multiplier,
            'emission_height_offset': emission_height_offset
        }
        
        return ray_origins, ray_direction, solar_info
    
    def _generate_ray_origins_for_angle(self, aabb, ray_direction, ray_spacing, 
                                      coverage_multiplier, emission_height_offset, debug):
        """
        Generate ray origin grid for angled rays (internal method)
        
        Parameters:
            aabb: Bounding box
            ray_direction: Ray direction vector
            ray_spacing: Ray spacing
            coverage_multiplier: Coverage multiplier
            emission_height_offset: Emission height offset
            debug: Debug flag
            
        Returns:
            ray_origins: Array of ray origins
        """
        # Get bounding box information
        min_bound = aabb.get_min_bound()
        max_bound = aabb.get_max_bound()
        extent = max_bound - min_bound
        center = (min_bound + max_bound) / 2
        
        # Calculate emission plane height
        plane_height = max_bound[2] + emission_height_offset
        
        # Calculate offset due to ray angle
        ground_center_z = min_bound[2] + extent[2] * 0.2  # Estimate ground center height
        vertical_distance = plane_height - ground_center_z
        
        # Calculate horizontal offset based on ray direction
        if abs(ray_direction[2]) > 1e-6:
            horizontal_offset_x = -ray_direction[0] * vertical_distance / ray_direction[2]
            horizontal_offset_y = -ray_direction[1] * vertical_distance / ray_direction[2]
        else:
            horizontal_offset_x = 0
            horizontal_offset_y = 0
        
        # Adjust emission area center
        adjusted_center_x = center[0] + horizontal_offset_x
        adjusted_center_y = center[1] + horizontal_offset_y
        
        # Calculate coverage area
        x_range = extent[0] * coverage_multiplier
        y_range = extent[1] * coverage_multiplier
        
        # Generate grid coordinates
        x_steps = int(x_range / ray_spacing) + 1
        y_steps = int(y_range / ray_spacing) + 1
        
        x_coords = np.linspace(adjusted_center_x - x_range/2, adjusted_center_x + x_range/2, x_steps)
        y_coords = np.linspace(adjusted_center_y - y_range/2, adjusted_center_y + y_range/2, y_steps)
        
        # Generate ray origins
        ray_origins = []
        for x in x_coords:
            for y in y_coords:
                origin = np.array([x, y, plane_height])
                ray_origins.append(origin)
        
        ray_origins = np.array(ray_origins)
        
        return ray_origins
    
    def get_solar_info_for_time_range(self, start_hour, end_hour, hour_step=1, 
                                    year=2024, month=6, day=21):
        """
        Get solar information for a time range
        
        Parameters:
            start_hour: Starting hour (0-23)
            end_hour: Ending hour (0-23)
            hour_step: Hour step size (default: 1)
            year: Year (default: 2024)
            month: Month (default: 6)
            day: Day (default: 21)
            
        Returns:
            time_series: List of solar information dictionaries
        """
        time_series = []
        
        for hour in range(start_hour, end_hour + 1, hour_step):
            elevation, azimuth = self.calculate_solar_position(year, month, day, hour, 0)
            
            if elevation > 0:  # Only record when sun is above horizon
                info = {
                    'time': f"{hour:02d}:00",
                    'hour': hour,
                    'elevation_deg': math.degrees(elevation),
                    'azimuth_deg': math.degrees(azimuth),
                    'elevation_rad': elevation,
                    'azimuth_rad': azimuth
                }
                time_series.append(info)
        
        return time_series
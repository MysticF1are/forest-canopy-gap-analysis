# Forest Canopy Gap Analysis System User Manual

## System Overview

The Forest Canopy Gap Analysis System is a point cloud-based light propagation and shadow analysis tool designed for analyzing forest light distribution, calculating shadow effects, and evaluating canopy gap fractions. The system employs ray tracing algorithms and Beer-Lambert light attenuation law, supporting multiple point cloud input formats.

## Dependency Installation

### Required Dependencies
```bash
pip install numpy
pip install scipy
pip install matplotlib
pip install open3d
pip install laspy
pip install PyQt5
```

### Recommended Dependencies
```bash
pip install pdal
pip install CSF
```

### Complete Installation Command
```bash
pip install numpy scipy matplotlib open3d laspy PyQt5 pdal CSF
```

## System Launch

### GUI Interface Launch
```bash
python main.py
```

### Command Line Interface
```python
from output import run_analysis

# Basic usage
output_dir = run_analysis(
    input_file="data.las",
    use_solar=True,
    ray_spacing=0.05,
    voxel_size=0.05,
    use_parallel=True,
    num_workers=4
)
```

## Supported File Formats

- **LAZ Format**: Compressed LiDAR data
- **LAS Format**: Standard LiDAR data
- **PCD Format**: Open3D point cloud files
- **PLY Format**: Polygon mesh files (experimental support)

## GUI Operation Guide

### File Loading Process
1. Click "Select File" button to choose point cloud file
2. System automatically executes preprocessing:
   - LAZ file decompression
   - Ground point detection
   - Vegetation layering (4 layers: ground, low vegetation, mid vegetation, canopy)
   - Data voxelization

### Parameter Configuration
- **Longitude**: Geographic longitude coordinate (degrees)
- **Latitude**: Geographic latitude coordinate (degrees)
- **Date/Time**: Reference time for solar position calculation

### Visualization Functions

#### Layer Display
- **Ground Layer**: Display ground layer points only
- **Ground + Low**: Display ground and low vegetation layers
- **Ground + Low + Mid**: Display ground, low vegetation, and mid vegetation layers
- **All Layers**: Display all vegetation layers

#### Projection Display
- **Ground Projection**: 2D projection of ground layer
- **All Points Projection**: 2D projection of all points (1/20 sampling)

#### Analysis Functions
- **Light Intensity Map**: Generate light intensity heatmap
- **Shadow Map**: Generate shadow distribution map
- **Export Analysis**: Export complete analysis report

## Core Algorithms

### Ray Tracing System
- Optimized AABB intersection detection
- Parallel computation acceleration support
- Beer-Lambert light attenuation law implementation
- Extinction coefficients: Low vegetation 7.13/m, Mid vegetation 5.75/m, Canopy 4.46/m

### Voxelization Processing
- Adaptive voxel size recommendation
- Multiple layering strategies: majority vote, ground priority, height priority
- Ground layer integrity validation

### Solar Position Calculation
- Real solar position calculation support
- Configurable date and time settings
- Automatic solar elevation and azimuth angle calculation

## Output Results Description

### Auto-generated Directory Structure
```
output_YYYYMMDD_HHMMSS/
├── analysis_report.txt
├── light_intensity_map.png
└── shadow_map.png
```

### Analysis Report Content
- Data summary statistics
- Voxelization information
- Light intensity analysis results
- Shadow distribution analysis
- Gap fraction calculation
- Processing parameter records

### Key Metrics Explanation
- **Light Intensity**: Range 0.0-1.0, where 1.0 indicates no attenuation
- **Gap Fraction**: Proportion of areas where light can penetrate to ground
- **Canopy Cover**: 1 - Gap Fraction
- **Success Rate**: Percentage of rays successfully reaching ground

## Performance Optimization

### Recommended Parameter Settings

#### Small-scale Data (<1M points)
```python
voxel_size = 0.03
ray_spacing = 0.025
batch_size = 500
```

#### Medium-scale Data (1-5M points)
```python
voxel_size = 0.05
ray_spacing = 0.05
batch_size = 1000
```

#### Large-scale Data (>5M points)
```python
voxel_size = 0.1
ray_spacing = 0.1
batch_size = 2000
```

### Parallel Computing Configuration
```python
use_parallel = True
num_workers = min(8, cpu_count())
```

## System Architecture

### Core Module Components
- **main.py**: GUI main interface
- **preprocess.py**: Point cloud preprocessing
- **voxelize.py**: Voxelization processing
- **solar_ray_system.py**: Ray generation system
- **optimized_ray_tracing.py**: Ray tracing engine
- **shadow_map.py**: Shadow calculation module
- **visualization_maps.py**: Visualization generation
- **output.py**: Result output module

### Data Flow
```
Raw Point Cloud → Preprocessing → Layering → Voxelization → Ray Generation → Ray Tracing → Result Visualization
```

## Caching Mechanism

The system implements intelligent cache management:
- Automatic caching of preprocessing results
- Voxelization result reuse
- Ray calculation result storage
- Automatic recalculation when parameters change

### Cache Directory Structure
```
cache/
├── [filename]_with_layers.pcd       # Processed point cloud with colors
├── [filename]_with_layers.ply       # PLY format output
├── [filename]_layers.npy           # Layer classification data
├── [filename]_voxelized.pcd        # Voxelized point cloud
├── [filename]_voxelized_layers.npy # Voxelized layer data
├── [filename]_voxel_info.json      # Voxelization statistics
└── [filename].las                  # Decompressed LAS (if input was LAZ)
```

For GUI usage, standard files are:
```
cache/
├── segment_with_layers.pcd         # Standard processed file
├── segment_layers.npy              # Standard layer file
└── segment_voxelized.pcd          # Standard voxelized file
```

## Troubleshooting

### Common Issues and Solutions

#### Dependency Library Issues
```
ImportError: No module named 'xxx'
```
Solution: Reinstall all libraries according to dependency list

#### File Format Issues
```
Error: Unsupported file format
```
Solution: Confirm file format is supported LAZ/LAS/PCD/PLY format

#### Memory Insufficient
```
MemoryError
```
Solution: Increase voxel_size parameter, reduce ray_spacing density

#### Slow Processing Speed
Solution: Enable parallel computing, adjust batch_size, use SSD storage

### Debugging Suggestions
1. Check console status logs
2. Review intermediate files in cache directory
3. Verify input data integrity
4. Adjust processing parameters and retry

## Technical Specifications

### Supported Data Scale
- Maximum point cloud size: No hard limit (memory constrained)
- Recommended single file size: < 1GB
- Processing speed: ~1M points/minute (standard configuration)

### Precision Specifications
- Spatial resolution: Determined by voxel_size (0.01-1.0 meters)
- Angular precision: 0.1 degrees
- Light intensity calculation precision: 3 decimal places

### System Requirements
- Python 3.7+
- Memory: 8GB+ recommended
- Storage: 3x data file space
- CPU: Multi-core recommended (parallel support)

## Extension Development

### Custom Parameters
```python
# Modify extinction coefficients
extinction_coefficients = {
    1: 7.13,  # Low vegetation
    2: 5.75,  # Mid vegetation
    3: 4.46   # Canopy
}

# Custom ray parameters
ray_params = {
    'ray_spacing': 0.05,
    'coverage_multiplier': 1.3,
    'emission_height_offset': 10.0
}
```

### Batch Processing Example
```python
import glob
from output import run_analysis

# Batch process multiple files
las_files = glob.glob("data/*.las")
for file_path in las_files:
    output_dir = run_analysis(
        input_file=file_path,
        use_solar=True,
        use_parallel=True
    )
    print(f"Processed: {file_path} -> {output_dir}")
```

## API Reference

### Main Functions

#### preprocess_las()
```python
def preprocess_las(input_path, cache_dir=None, progress_callback=None):
    """
    Main preprocessing interface
    
    Parameters:
        input_path: Path to input LAZ/LAS file
        cache_dir: Cache directory (default: ./cache)
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with processed file paths and statistics
    """
```

#### run_analysis()
```python
def run_analysis(input_file, output_dir=None, use_solar=False, 
                 solar_params=None, ray_spacing=0.05, voxel_size=0.05,
                 use_parallel=True, num_workers=4):
    """
    Run complete light and shadow analysis
    
    Parameters:
        input_file: Input point cloud file path
        output_dir: Output directory (None = auto-generate)
        use_solar: Use solar rays (True) or vertical rays (False)
        solar_params: Solar parameters dict
        ray_spacing: Ray spacing in meters
        voxel_size: Voxel size for processing
        use_parallel: Use parallel computation
        num_workers: Number of parallel workers
        
    Returns:
        output_dir: Path to output directory
    """
```

### Class References

#### OptimizedRayTracingSystem
```python
class OptimizedRayTracingSystem:
    def __init__(self, voxel_size=0.05, debug=False)
    def load_layered_point_cloud(self, points, layers)
    def compute_ground_light_distribution(self, ray_origins, ray_direction, ...)
```

#### SolarRaycastingSystem
```python
class SolarRaycastingSystem:
    def __init__(self, latitude=0.0, longitude=0.0, timezone_offset=0)
    def generate_solar_rays(self, aabb, year, month, day, hour, minute, ...)
    def generate_vertical_test_rays(self, aabb, ray_spacing=0.025, ...)
```

## Configuration Examples

### Solar Analysis Configuration
```python
solar_params = {
    'year': 2024,
    'month': 6,      # June
    'day': 21,       # Summer solstice
    'hour': 12,      # Noon
    'minute': 0,
    'latitude': 51.5074,   # London
    'longitude': -0.1278
}

output_dir = run_analysis(
    input_file="forest_data.las",
    use_solar=True,
    solar_params=solar_params,
    ray_spacing=0.05,
    use_parallel=True
)
```

### High-Resolution Analysis
```python
output_dir = run_analysis(
    input_file="detailed_forest.las",
    ray_spacing=0.02,      # High resolution
    voxel_size=0.03,       # Fine voxels
    use_parallel=True,
    num_workers=8
)
```

## Citation and License

This system is developed based on open-source libraries, mainly depending on:
- Open3D: Point cloud processing
- NumPy/SciPy: Numerical computation
- Matplotlib: Visualization
- PyQt5: Graphical interface
- LASPy: LAS file processing

Please ensure compliance with relevant open-source licenses when using.
```

# Forest Canopy Gap Analysis System

**MSc Advanced Computer Science Research Project (2024-2025)**

Point cloud-based forest light analysis using ray tracing and Beer-Lambert law.

## Features

- Ray tracing with solar position modeling
- Multi-format support (LAZ/LAS/PCD/PLY)
- Parallel processing for large datasets
- GUI and command line interfaces
- Automatic vegetation layering

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib open3d laspy PyQt5 pdal CSF

# Run GUI
python main.py

# Command line
from output import run_analysis
output_dir = run_analysis("data.las", use_solar=True, use_parallel=True)
```

## Output

- Light intensity heatmaps
- Shadow distribution maps
- Gap fraction analysis
- Statistical reports

## Requirements

- Python 3.7+
- 8GB+ RAM recommended
- Multi-core CPU for parallel processing

## Documentation

See [USER_GUIDE_EN.md](USER_GUIDE_EN.md) for complete documentation.

## License

GPL v3.0 - see [LICENSE](LICENSE) file.

## Citation

```bibtex
@software{Li_Forest_Canopy_Gap_2025,
author = {Li, Zhaohao},
title = {{Forest Canopy Gap Analysis System}},
url = {https://github.com/MysticF1are/forest-canopy-gap-analysis},
version = {0.1.0},
year = {2025}
}
```

# main.py - Complete Main GUI with File Processing

import sys
import os
import numpy as np
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QTextEdit, QGroupBox, QGridLayout, QLineEdit,
                            QDateTimeEdit, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QDateTime
from PyQt5.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import open3d as o3d

# Import processing modules
from preprocess import preprocess_las
from voxelize import voxelize_with_layers
from solar_ray_system import SolarRaycastingSystem
from optimized_ray_tracing import OptimizedRayTracingSystem
from shadow_map import ShadowMapGenerator
from visualization_maps import show_result_heatmap_only, calculate_gap_fraction, display_intensity_distribution
from output import run_analysis


class FileProcessingThread(QThread):
    """Thread for file preprocessing"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            file_path = Path(self.file_path)
            file_ext = file_path.suffix.lower()
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            
            # Determine processing based on file type
            if file_ext == '.laz':
                self.progress.emit("Processing LAZ file...")
                self.progress.emit("Decompressing LAZ to LAS...")
                # Preprocess will handle LAZ->LAS conversion
                result = preprocess_las(
                    str(file_path),
                    cache_dir="cache",
                    progress_callback=lambda msg: self.progress.emit(msg)
                )
                
            elif file_ext == '.las':
                self.progress.emit("Processing LAS file...")
                result = preprocess_las(
                    str(file_path),
                    cache_dir="cache",
                    progress_callback=lambda msg: self.progress.emit(msg)
                )
                
            elif file_ext == '.pcd':
                self.progress.emit("Loading PCD file...")
                # Check if corresponding layers file exists
                layers_file = cache_dir / f"{file_path.stem}_layers.npy"
                if not layers_file.exists():
                    # Look for any layers file in cache
                    layers_file = cache_dir / "segment_layers.npy"
                
                if layers_file.exists():
                    result = {
                        'pcd': str(file_path),
                        'layers': str(layers_file),
                        'ready': True
                    }
                else:
                    self.error.emit("No layer information found for PCD file. Please load a LAS/LAZ file first.")
                    return
                    
            elif file_ext == '.ply':
                self.error.emit("PLY format support is not yet implemented. Please use LAS, LAZ, or PCD files.")
                return
                
            else:
                self.error.emit(f"Unsupported file format: {file_ext}")
                return
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))

class CalculationThread(QThread):
    """Thread for heavy calculations (heatmap and shadow)"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, calc_type, data):
        super().__init__()
        self.calc_type = calc_type
        self.data = data
        
    def run(self):
        try:
            # Import necessary modules
            from voxelize import voxelize_with_layers
            from solar_ray_system import SolarRaycastingSystem
            from optimized_ray_tracing import OptimizedRayTracingSystem
            from shadow_map import ShadowMapGenerator
            import open3d as o3d
            
            # Check if we need to compute voxels and rays
            if self.data.get('voxel_data_cache') is None or self.data.get('ray_data_cache') is None:
                # Step 1: Voxelize
                self.progress.emit("Step 1/4: Voxelizing point cloud...")
                voxel_points, voxel_layers, voxel_info = voxelize_with_layers(
                    self.data['points'],
                    self.data['layers'],
                    voxel_size=0.05,
                    strategy='ground_priority'
                )
                self.progress.emit(f"Voxelized to {len(voxel_points):,} points")
                
                # Step 2: Generate rays
                self.progress.emit("Step 2/4: Generating rays...")
                min_bound = voxel_points.min(axis=0)
                max_bound = voxel_points.max(axis=0)
                aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)

                # Use equatorial position for near-vertical sun rays
                solar_system = SolarRaycastingSystem(
                    latitude=self.data.get('latitude', 0.0),
                    longitude=self.data.get('longitude', 0.0),
                    timezone_offset=0
                )

                # Generate solar rays
                datetime = self.data.get('datetime')
                if datetime:
                    ray_origins, ray_direction, ray_info = solar_system.generate_solar_rays(
                        aabb,
                        year=datetime.year,
                        month=datetime.month,
                        day=datetime.day,
                        hour=datetime.hour,
                        minute=datetime.minute,
                        ray_spacing=0.05,
                        coverage_multiplier=1.3,
                        emission_height_offset=10.0
                    )
                else:
                    # Default: equinox noon
                    ray_origins, ray_direction, ray_info = solar_system.generate_solar_rays(
                        aabb,
                        year=2024,
                        month=3,
                        day=21,
                        hour=12,
                        minute=0,
                        ray_spacing=0.05,
                        coverage_multiplier=1.3,
                        emission_height_offset=10.0
                    )

                self.progress.emit(f"Generated {len(ray_origins):,} rays")

                if 'elevation_deg' in ray_info:
                    self.progress.emit(f"Sun elevation: {ray_info['elevation_deg']:.1f}°")
            else:
                # Use cached data
                self.progress.emit("Using cached voxel and ray data...")
                voxel_points, voxel_layers, voxel_info = self.data['voxel_data_cache']
                ray_origins, ray_direction, ray_info = self.data['ray_data_cache']
            
            if self.calc_type == 'heatmap':
                # Check if we have cached light distribution
                if self.data.get('light_distribution_cache') is not None:
                    self.progress.emit("Using cached light intensity results...")
                    distribution = self.data['light_distribution_cache']
                else:
                    # Step 3: Calculate light intensity
                    self.progress.emit("Step 3/4: Calculating light intensity...")
                    tracer = OptimizedRayTracingSystem(voxel_size=0.05)
                    tracer.load_layered_point_cloud(voxel_points, voxel_layers)
                    
                    distribution = tracer.compute_ground_light_distribution(
                        ray_origins, ray_direction,
                        batch_size=1000,
                        use_parallel=True,
                        num_workers=4
                    )
                    self.progress.emit(f"Light calculation complete: {len(distribution['positions']):,} hits")
                
            elif self.calc_type == 'shadow':
                # Check if we have cached shadow distribution
                if self.data.get('shadow_distribution_cache') is not None:
                    self.progress.emit("Using cached shadow results...")
                    distribution = self.data['shadow_distribution_cache']
                else:
                    # Step 3: Calculate shadow
                    self.progress.emit("Step 3/4: Calculating shadow distribution...")
                    shadow_gen = ShadowMapGenerator(voxel_size=0.3)
                    shadow_gen.load_layered_point_cloud(voxel_points, voxel_layers)
                    
                    distribution = shadow_gen.compute_shadow_map(
                        ray_origins, ray_direction,
                        batch_size=2000
                    )
                    self.progress.emit(f"Shadow calculation complete: {len(distribution['positions']):,} hits")
                    
            elif self.calc_type == 'both':
                # Calculate both for export
                self.progress.emit("Calculating light intensity and shadow maps...")
                
                # Light intensity
                if self.data.get('light_distribution_cache') is not None:
                    self.progress.emit("Using cached light intensity...")
                    light_distribution = self.data['light_distribution_cache']
                else:
                    self.progress.emit("Calculating light intensity...")
                    tracer = OptimizedRayTracingSystem(voxel_size=0.05)
                    tracer.load_layered_point_cloud(voxel_points, voxel_layers)
                    light_distribution = tracer.compute_ground_light_distribution(
                        ray_origins, ray_direction,
                        batch_size=1000,
                        use_parallel=True,
                        num_workers=4
                    )
                
                # Shadow map
                if self.data.get('shadow_distribution_cache') is not None:
                    self.progress.emit("Using cached shadow map...")
                    shadow_distribution = self.data['shadow_distribution_cache']
                else:
                    self.progress.emit("Calculating shadow map...")
                    shadow_gen = ShadowMapGenerator(voxel_size=0.3)
                    shadow_gen.load_layered_point_cloud(voxel_points, voxel_layers)
                    shadow_distribution = shadow_gen.compute_shadow_map(
                        ray_origins, ray_direction,
                        batch_size=2000
                    )
                
                # Return both distributions
                result = {
                    'light_distribution': light_distribution,
                    'shadow_distribution': shadow_distribution,
                    'voxel_points': voxel_points,
                    'voxel_layers': voxel_layers,
                    'voxel_info': voxel_info,
                    'ray_data': (ray_origins, ray_direction, ray_info),
                    'calc_type': 'both'
                }
                
                self.finished.emit(result)
                return
            
            # Step 4: Prepare result
            self.progress.emit("Step 4/4: Preparing visualization...")
            result = {
                'distribution': distribution,
                'voxel_points': voxel_points,
                'voxel_layers': voxel_layers,
                'voxel_info': voxel_info,
                'ray_data': (ray_origins, ray_direction, ray_info),
                'calc_type': self.calc_type
            }
            
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class VisualizationThread(QThread):
    """Thread for visualization tasks"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, task_type, data):
        super().__init__()
        self.task_type = task_type
        self.data = data
        
    def run(self):
        try:
            result = {}
            
            if self.task_type == 'show_layers':
                # Show specific layers
                layers_to_show = self.data['layers_to_show']
                points = self.data['points']
                layers = self.data['layers']
                
                mask = np.isin(layers, layers_to_show)
                filtered_points = points[mask]
                filtered_layers = layers[mask]
                
                result = {
                    'points': filtered_points,
                    'layers': filtered_layers,
                    'title': self.data['title']
                }
                
            elif self.task_type == 'projection':
                # Create projection
                points = self.data['points']
                sample_rate = self.data.get('sample_rate', 1)
                
                if sample_rate > 1:
                    points = points[::sample_rate]
                
                projected = points.copy()
                projected[:, 2] = 0
                
                result = {
                    'points': projected,
                    'title': self.data['title']
                }
                
            elif self.task_type == 'output':
                self.progress.emit("Generating analysis report...")
                # Generate report based on cached data
                result = {'report_generated': True}
                
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.pcd_file = None
        self.layers_file = None
        self.points = None
        self.colors = None
        self.layers = None
        self.current_thread = None
        
        # Cache for calculation results
        self.light_distribution_cache = None
        self.shadow_distribution_cache = None
        self.voxel_data_cache = None  # (voxel_points, voxel_layers, voxel_info)
        self.ray_data_cache = None    # (ray_origins, ray_direction, ray_info)
        self.last_calc_params = None  # Store parameters to check if recalc needed
        
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Forest Canopy Gap Analysis System')
        self.setGeometry(100, 100, 1200, 800)
        
        # Set style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 4px;
                font-size: 14px;
                font-weight: bold;
                min-height: 35px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-size: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Main widget
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left control panel
        left_panel = QWidget()
        left_panel.setMaximumWidth(400)
        left_layout = QVBoxLayout(left_panel)
        
        # File selection area
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()
        
        self.file_button = QPushButton("Select File")
        self.file_button.clicked.connect(self.select_file)
        self.file_label = QLabel("No file selected")
        self.file_label.setWordWrap(True)
        
        file_layout.addWidget(self.file_button)
        file_layout.addWidget(self.file_label, 1)
        file_group.setLayout(file_layout)
        
        # Parameter input area
        param_group = QGroupBox("Parameters")
        param_layout = QGridLayout()
        param_layout.setSpacing(10)
        
        param_layout.addWidget(QLabel("Longitude:"), 0, 0)
        self.longitude_input = QLineEdit("0.0")  # Equatorial longitude
        self.longitude_input.setPlaceholderText("Equator: 0.0°")
        param_layout.addWidget(self.longitude_input, 0, 1)
        
        param_layout.addWidget(QLabel("Latitude:"), 1, 0)
        self.latitude_input = QLineEdit("0.0")  # Equatorial latitude
        self.latitude_input.setPlaceholderText("Equator: 0.0°")
        param_layout.addWidget(self.latitude_input, 1, 1)
        
        param_layout.addWidget(QLabel("Date/Time:"), 2, 0)
        self.datetime_input = QDateTimeEdit()
        # Set to March 21, 2024 12:00 (Spring equinox noon)
        equinox_datetime = QDateTime(2024, 3, 21, 12, 0, 0)
        self.datetime_input.setDateTime(equinox_datetime)
        self.datetime_input.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.datetime_input.setCalendarPopup(True)
        param_layout.addWidget(self.datetime_input, 2, 1)
        
        # Add info label
        info_label = QLabel("Default: Equinox noon for vertical rays")
        info_label.setStyleSheet("color: #666; font-size: 11px; font-style: italic;")
        param_layout.addWidget(info_label, 3, 0, 1, 2)
        
        param_group.setLayout(param_layout)
        
        # Function buttons area
        button_group = QGroupBox("Visualization Functions")
        button_layout = QGridLayout()
        button_layout.setSpacing(10)
        
        # Create 9 function buttons
        self.buttons = {}
        button_configs = [
            ("Ground Layer", "show_ground", 0, 0, "#8B4513"),
            ("Ground + Low", "show_ground_low", 0, 1, "#90EE90"),
            ("Ground + Low + Mid", "show_ground_low_mid", 0, 2, "#228B22"),
            ("All Layers", "show_all", 1, 0, "#006400"),
            ("Ground Projection", "project_ground", 1, 1, "#4169E1"),
            ("All Points Projection", "project_all", 1, 2, "#1E90FF"),
            ("Light Intensity Map", "heatmap", 2, 0, "#FF4500"),
            ("Shadow Map", "shadow", 2, 1, "#696969"),
            ("Export Analysis", "export", 2, 2, "#9370DB")
        ]
        
        for name, key, row, col, color in button_configs:
            btn = QPushButton(name)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, k=key: self.execute_function(k))
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {color};
                }}
                QPushButton:hover {{
                    background-color: {color}dd;
                }}
                QPushButton:disabled {{
                    background-color: #cccccc;
                }}
            """)
            self.buttons[key] = btn
            button_layout.addWidget(btn, row, col)
        
        button_group.setLayout(button_layout)
        
        # Status display area
        status_group = QGroupBox("Status Log")
        status_layout = QVBoxLayout()
        
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        self.status_text.setMaximumHeight(200)
        self.add_status("System initialized. Please select a point cloud file.")
        
        status_layout.addWidget(self.status_text)
        status_group.setLayout(status_layout)
        
        # Assemble left panel
        left_layout.addWidget(file_group)
        left_layout.addWidget(param_group)
        left_layout.addWidget(button_group)
        left_layout.addWidget(status_group)
        left_layout.addStretch()
        
        # Right visualization area
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.viz_title = QLabel("Visualization Area")
        self.viz_title.setAlignment(Qt.AlignCenter)
        self.viz_title.setStyleSheet("""
            QLabel {
                font-size: 18px;
                font-weight: bold;
                padding: 10px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        self.figure = Figure(figsize=(10, 8))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("""
            background-color: white;
            border: 1px solid #ddd;
            border-radius: 5px;
        """)
        
        right_layout.addWidget(self.viz_title)
        right_layout.addWidget(self.canvas, 1)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel, 1)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def get_current_params(self):
        """Get current parameter configuration"""
        try:
            latitude = float(self.latitude_input.text().strip()) if self.latitude_input.text().strip() else 0.0
            longitude = float(self.longitude_input.text().strip()) if self.longitude_input.text().strip() else 0.0
        except:
            latitude = 0.0
            longitude = 0.0
        
        datetime = self.datetime_input.dateTime().toPyDateTime()
        
        return {
            'latitude': latitude,
            'longitude': longitude,
            'datetime': datetime.isoformat(),
            'file': self.current_file
        }
    
    def params_changed(self):
        """Check if parameters have changed since last calculation"""
        current_params = self.get_current_params()
        return current_params != self.last_calc_params
    
    def clear_cache(self):
        """Clear all cached calculation results"""
        self.light_distribution_cache = None
        self.shadow_distribution_cache = None
        self.voxel_data_cache = None
        self.ray_data_cache = None
        self.last_calc_params = None
        self.add_status("Cache cleared - next operation will recalculate")
    
    def select_file(self):
        """Open file dialog to select point cloud file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Point Cloud File",
            "",
            "Point Cloud Files (*.laz *.las *.pcd *.ply);;All Files (*.*)"
        )
        
        if file_path:
            self.current_file = file_path
            self.file_label.setText(os.path.basename(file_path))
            self.add_status(f"File selected: {os.path.basename(file_path)}")
            
            # Clear cache when new file is loaded
            self.clear_cache()
            
            # Disable buttons during processing
            for btn in self.buttons.values():
                btn.setEnabled(False)
            
            # Start file processing thread
            self.process_file(file_path)
    
    def process_file(self, file_path):
        """Process selected file based on its type"""
        self.add_status(f"Processing file: {os.path.basename(file_path)}")
        
        # Create and start processing thread
        self.file_thread = FileProcessingThread(file_path)
        self.file_thread.progress.connect(self.add_status)
        self.file_thread.finished.connect(self.on_file_processed)
        self.file_thread.error.connect(self.on_processing_error)
        self.file_thread.start()
    
    def on_file_processed(self, result):
        """Handle successful file processing"""
        self.add_status("File processing completed!")
        
        # Determine final PCD and layers files
        if 'pcd' in result:
            self.pcd_file = result['pcd']
            self.layers_file = result.get('layers', 'cache/segment_layers.npy')
        else:
            # For LAZ/LAS processing, use standard names
            self.pcd_file = 'cache/segment_with_layers.pcd'
            self.layers_file = 'cache/segment_layers.npy'
        
        # Load the processed data
        self.load_processed_data()
        
        # Enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        self.statusBar().showMessage("Ready to visualize")
    
    def on_processing_error(self, error_msg):
        """Handle file processing error"""
        self.add_status(f"Error: {error_msg}")
        QMessageBox.critical(self, "Processing Error", error_msg)
        self.statusBar().showMessage("Error occurred")
    
    def load_processed_data(self):
        """Load processed PCD and layer data"""
        try:
            # Load PCD file
            if os.path.exists(self.pcd_file):
                pcd = o3d.io.read_point_cloud(self.pcd_file)
                self.points = np.asarray(pcd.points)
                self.colors = np.asarray(pcd.colors) if pcd.has_colors() else None
                
            # Load layers file
            if os.path.exists(self.layers_file):
                self.layers = np.load(self.layers_file)
                
            self.add_status(f"Loaded {len(self.points):,} points with {len(np.unique(self.layers))} layers")
            
        except Exception as e:
            self.add_status(f"Error loading data: {str(e)}")
    
    def execute_function(self, function_key):
        """Execute selected visualization function"""
        if self.points is None or self.layers is None:
            QMessageBox.warning(self, "No Data", "Please load a file first!")
            return
        
        self.add_status(f"Executing: {function_key}")
        
        if function_key == 'show_ground':
            self.show_layers([0], "Ground Layer")
            
        elif function_key == 'show_ground_low':
            self.show_layers([0, 1], "Ground + Low Vegetation")
            
        elif function_key == 'show_ground_low_mid':
            self.show_layers([0, 1, 2], "Ground + Low + Mid Vegetation")
            
        elif function_key == 'show_all':
            self.show_layers([0, 1, 2, 3], "All Layers")
            
        elif function_key == 'project_ground':
            self.show_projection(layer=0, title="Ground Layer Projection")
            
        elif function_key == 'project_all':
            self.show_projection(layer='all', sample_rate=20, title="All Points Projection (1/20)")
            
        elif function_key == 'heatmap':
            self.show_heatmap()
            
        elif function_key == 'shadow':
            self.show_shadow()
            
        elif function_key == 'export':
            self.export_analysis()
    
    def show_layers(self, layers_to_show, title):
        """Display specific layers in 3D"""
        mask = np.isin(self.layers, layers_to_show)
        filtered_points = self.points[mask]
        filtered_layers = self.layers[mask]
        
        # Update visualization
        self.figure.clear()
        ax = self.figure.add_subplot(111, projection='3d')
        
        # Color by layer
        colors_map = {0: 'brown', 1: 'lightgreen', 2: 'green', 3: 'darkgreen'}
        total_displayed = 0
        for layer in layers_to_show:
            layer_mask = filtered_layers == layer
            if np.any(layer_mask):
                layer_points = filtered_points[layer_mask]
                # Subsample for performance
                sampled_points = layer_points[::20]
                if len(sampled_points) > 0:
                    ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2],
                            c=colors_map.get(layer, 'gray'), s=0.1, alpha=0.5)
                    total_displayed += len(sampled_points)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f"{title}\n{total_displayed:,} points displayed")
        
        self.viz_title.setText(title)
        self.canvas.draw()
    
    def show_projection(self, layer='all', sample_rate=1, title="Projection"):
        """Display 2D projection"""
        if layer == 'all':
            points_to_project = self.points[::sample_rate]
        else:
            mask = self.layers == layer
            points_to_project = self.points[mask]
        
        # Create projection
        projected = points_to_project.copy()
        projected[:, 2] = 0
        
        # Update visualization
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.scatter(projected[:, 0], projected[:, 1], s=0.05, c='darkblue', alpha=0.3)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_aspect('equal')
        ax.set_title(f"{title}\n{len(projected):,} points")
        
        self.viz_title.setText(title)
        self.canvas.draw()
    
    def show_heatmap(self):
        """Display light intensity heatmap"""
        if self.points is None or self.layers is None:
            QMessageBox.warning(self, "No Data", "Please load a file first!")
            return
        
        # Check if we have cached results and parameters haven't changed
        if self.light_distribution_cache is not None and not self.params_changed():
            self.add_status("Using cached light intensity results")
            # Display cached results directly
            self.display_light_intensity()
            return
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Calculating light intensity...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # Disable buttons
        for btn in self.buttons.values():
            btn.setEnabled(False)
        
        # Prepare data
        try:
            latitude_text = self.latitude_input.text().strip()
            longitude_text = self.longitude_input.text().strip()
            latitude = float(latitude_text) if latitude_text else 0.0
            longitude = float(longitude_text) if longitude_text else 0.0
        except:
            latitude = 0.0
            longitude = 0.0
            self.add_status("Using default equatorial coordinates (0.0, 0.0)")
        
        data = {
            'points': self.points,
            'layers': self.layers,
            'latitude': latitude,
            'longitude': longitude,
            'datetime': self.datetime_input.dateTime().toPyDateTime(),
            'use_solar': True,
            'voxel_data_cache': self.voxel_data_cache,
            'ray_data_cache': self.ray_data_cache,
            'light_distribution_cache': self.light_distribution_cache
        }
        
        # Create and start calculation thread
        self.calc_thread = CalculationThread('heatmap', data)
        self.calc_thread.progress.connect(self.update_progress)
        self.calc_thread.finished.connect(self.on_heatmap_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        
        # Connect cancel button
        self.progress_dialog.canceled.connect(self.cancel_calculation)
        
        self.calc_thread.start()

    def show_shadow(self):
        """Display shadow map"""
        if self.points is None or self.layers is None:
            QMessageBox.warning(self, "No Data", "Please load a file first!")
            return
        
        # Check if we have cached results and parameters haven't changed
        if self.shadow_distribution_cache is not None and not self.params_changed():
            self.add_status("Using cached shadow results")
            # Display cached results directly
            self.display_shadow_map()
            return
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog("Calculating shadow distribution...", "Cancel", 0, 0, self)
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setValue(0)
        self.progress_dialog.show()
        
        # Disable buttons
        for btn in self.buttons.values():
            btn.setEnabled(False)
        
        # Prepare data
        try:
            latitude_text = self.latitude_input.text().strip()
            longitude_text = self.longitude_input.text().strip()
            latitude = float(latitude_text) if latitude_text else 0.0
            longitude = float(longitude_text) if longitude_text else 0.0
        except:
            latitude = 0.0
            longitude = 0.0
            self.add_status("Using default equatorial coordinates (0.0, 0.0)")
        
        data = {
            'points': self.points,
            'layers': self.layers,
            'latitude': latitude,
            'longitude': longitude,
            'datetime': self.datetime_input.dateTime().toPyDateTime(),
            'use_solar': True,
            'voxel_data_cache': self.voxel_data_cache,
            'ray_data_cache': self.ray_data_cache,
            'shadow_distribution_cache': self.shadow_distribution_cache
        }
        
        # Create and start calculation thread
        self.calc_thread = CalculationThread('shadow', data)
        self.calc_thread.progress.connect(self.update_progress)
        self.calc_thread.finished.connect(self.on_shadow_finished)
        self.calc_thread.error.connect(self.on_calculation_error)
        
        # Connect cancel button
        self.progress_dialog.canceled.connect(self.cancel_calculation)
        
        self.calc_thread.start()

    def update_progress(self, message):
        """Update progress dialog"""
        self.add_status(message)
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setLabelText(message)

    def cancel_calculation(self):
        """Cancel current calculation"""
        if hasattr(self, 'calc_thread') and self.calc_thread.isRunning():
            self.calc_thread.terminate()
            self.calc_thread.wait()
            self.add_status("Calculation cancelled")
            
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()

    def on_heatmap_finished(self, result):
        """Handle heatmap calculation completion"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        # Cache the results
        self.light_distribution_cache = result['distribution']
        self.voxel_data_cache = (result['voxel_points'], result['voxel_layers'], result.get('voxel_info'))
        self.ray_data_cache = result.get('ray_data')
        self.last_calc_params = self.get_current_params()
        
        self.add_status("Light intensity map completed and cached!")
        
        # Display the heatmap
        self.display_light_intensity()
        
        self.statusBar().showMessage("Light intensity map displayed")
    
    def display_light_intensity(self):
        """Display cached light intensity results"""
        if self.light_distribution_cache is None:
            return
        
        # Get voxel data
        if self.voxel_data_cache:
            voxel_points, voxel_layers, _ = self.voxel_data_cache
        else:
            voxel_points, voxel_layers = self.points, self.layers
        
        display_intensity_distribution(
            voxel_points,
            voxel_layers,
            self.light_distribution_cache,
            self.points,
            self.layers
        )
        
        # Show statistics
        intensities = self.light_distribution_cache['intensities']
        self.add_status(f"Statistics:")
        self.add_status(f"  - Rays hit ground: {len(intensities):,}")
        self.add_status(f"  - Mean intensity: {np.mean(intensities):.3f}")
        self.add_status(f"  - Intensity range: [{np.min(intensities):.3f}, {np.max(intensities):.3f}]")

    def on_shadow_finished(self, result):
        """Handle shadow calculation completion"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        # Cache the results
        self.shadow_distribution_cache = result['distribution']
        self.voxel_data_cache = (result['voxel_points'], result['voxel_layers'], result.get('voxel_info'))
        self.ray_data_cache = result.get('ray_data')
        self.last_calc_params = self.get_current_params()
        
        self.add_status("Shadow map completed and cached!")
        
        # Display the shadow map
        self.display_shadow_map()
        
        self.statusBar().showMessage("Shadow map displayed")
    
    def display_shadow_map(self):
        """Display cached shadow map results"""
        if self.shadow_distribution_cache is None:
            return
        
        # Get voxel data
        if self.voxel_data_cache:
            voxel_points, voxel_layers, _ = self.voxel_data_cache
        else:
            voxel_points, voxel_layers = self.points, self.layers
        
        display_intensity_distribution(
            voxel_points,
            voxel_layers,
            self.shadow_distribution_cache,
            self.points,
            self.layers
        )
        
        # Calculate and show gap fraction
        gap_stats = calculate_gap_fraction(self.shadow_distribution_cache)
        if gap_stats:
            self.add_status(f"Gap Fraction Analysis:")
            self.add_status(f"  - Gap Fraction: {gap_stats['gap_fraction']:.3f} ({gap_stats['gap_fraction']*100:.1f}%)")
            self.add_status(f"  - Canopy Cover: {gap_stats['canopy_cover']:.3f} ({gap_stats['canopy_cover']*100:.1f}%)")
            self.add_status(f"  - Illuminated points: {gap_stats['gap_points']:,}")
            self.add_status(f"  - Shadowed points: {gap_stats['shadowed_points']:,}")

    def on_calculation_error(self, error_msg):
        """Handle calculation error"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        self.add_status(f"Error: {error_msg}")
        QMessageBox.critical(self, "Calculation Error", f"An error occurred:\n{error_msg}")
        self.statusBar().showMessage("Error occurred")
    
    def export_analysis(self):
        """Export complete analysis with caching"""
        reply = QMessageBox.question(self, "Export Analysis", 
                                    "This will run analysis and save all results.\nContinue?",
                                    QMessageBox.Yes | QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.add_status("Starting analysis export...")
            
            # Show progress dialog
            self.progress_dialog = QProgressDialog("Running analysis...", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowTitle("Exporting Analysis")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.setValue(0)
            self.progress_dialog.show()
            
            # Disable buttons
            for btn in self.buttons.values():
                btn.setEnabled(False)
            
            # Get parameters
            try:
                latitude_text = self.latitude_input.text().strip()
                longitude_text = self.longitude_input.text().strip()
                latitude = float(latitude_text) if latitude_text else 0.0
                longitude = float(longitude_text) if longitude_text else 0.0
            except ValueError:
                latitude = 0.0
                longitude = 0.0
            
            datetime = self.datetime_input.dateTime().toPyDateTime()
            
            # Prepare data with all caches
            data = {
                'points': self.points,
                'layers': self.layers,
                'latitude': latitude,
                'longitude': longitude,
                'datetime': datetime,
                'use_solar': True,
                'voxel_data_cache': self.voxel_data_cache,
                'ray_data_cache': self.ray_data_cache,
                'light_distribution_cache': self.light_distribution_cache,
                'shadow_distribution_cache': self.shadow_distribution_cache
            }
            
            # Create calculation thread for both if needed
            self.export_thread = CalculationThread('both', data)
            self.export_thread.progress.connect(self.update_progress)
            self.export_thread.finished.connect(self.on_export_finished)
            self.export_thread.error.connect(self.on_export_error)
            
            # Connect cancel button
            self.progress_dialog.canceled.connect(self.cancel_export)
            
            self.export_thread.start()
    
    def cancel_export(self):
        """Cancel export"""
        if hasattr(self, 'export_thread') and self.export_thread.isRunning():
            self.export_thread.terminate()
            self.export_thread.wait()
            self.add_status("Export cancelled")
        
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
    
    def on_export_finished(self, result):
        """Handle export completion"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        # Update caches with results
        if 'light_distribution' in result:
            self.light_distribution_cache = result['light_distribution']
        if 'shadow_distribution' in result:
            self.shadow_distribution_cache = result['shadow_distribution']
        if 'voxel_points' in result:
            self.voxel_data_cache = (result['voxel_points'], result['voxel_layers'], result.get('voxel_info'))
        if 'ray_data' in result:
            self.ray_data_cache = result['ray_data']
        
        self.last_calc_params = self.get_current_params()
        
        # Generate and save report
        self.generate_export_report(result)
    
    def generate_export_report(self, result):
        """Generate and save analysis report"""
        from datetime import datetime
        import os
        from pathlib import Path
        
        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"output_{timestamp}")
        output_dir.mkdir(exist_ok=True)
        
        # File paths in output directory
        report_filename = output_dir / "analysis_report.txt"
        light_map_filename = output_dir / "light_intensity_map.png"
        shadow_map_filename = output_dir / "shadow_map.png"
        
        # Create report content
        report = []
        report.append("="*60)
        report.append("FOREST CANOPY GAP ANALYSIS REPORT")
        report.append("="*60)
        report.append("")
        
        # Basic information
        report.append("Analysis Information:")
        report.append(f"  Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Working Directory: {os.getcwd()}")
        report.append(f"  Output Directory: {output_dir}")
        report.append(f"  Input File: {self.current_file if self.current_file else 'N/A'}")
        report.append("")
        
        # Data summary
        if self.points is not None and self.layers is not None:
            report.append("Data Summary:")
            report.append(f"  Original Points: {len(self.points):,}")
            unique_layers = np.unique(self.layers)
            report.append(f"  Layers: {len(unique_layers)} layers")
            for layer in unique_layers:
                count = np.sum(self.layers == layer)
                percentage = count / len(self.layers) * 100
                layer_names = {0: "Ground", 1: "Low Vegetation", 2: "Mid Vegetation", 3: "Canopy"}
                name = layer_names.get(layer, f"Layer {layer}")
                report.append(f"    - {name}: {count:,} ({percentage:.1f}%)")
            report.append("")
        
        # Voxelization info
        if self.voxel_data_cache:
            voxel_points, voxel_layers, voxel_info = self.voxel_data_cache
            report.append("Voxelization:")
            report.append(f"  Voxel Count: {len(voxel_points):,}")
            if voxel_info:
                report.append(f"  Compression Ratio: {voxel_info.get('compression_ratio', 'N/A'):.1f}:1")
                report.append(f"  Voxel Size: {voxel_info.get('voxel_size', 0.05):.3f} m")
            report.append("")
        
        # Light intensity results
        if self.light_distribution_cache:
            intensities = self.light_distribution_cache['intensities']
            report.append("Light Intensity Analysis:")
            report.append(f"  Successful Rays: {len(intensities):,}")
            if 'metadata' in self.light_distribution_cache:
                meta = self.light_distribution_cache['metadata']
                report.append(f"  Total Rays: {meta.get('total_rays', 'N/A'):,}")
                report.append(f"  Success Rate: {meta.get('success_rate', 'N/A'):.1f}%")
            
            if len(intensities) > 0:
                report.append(f"  Mean Intensity: {np.mean(intensities):.3f}")
                report.append(f"  Intensity Range: [{np.min(intensities):.3f}, {np.max(intensities):.3f}]")
                report.append(f"  Standard Deviation: {np.std(intensities):.3f}")
            report.append("")
        
        # Shadow analysis results
        if self.shadow_distribution_cache:
            shadow_intensities = self.shadow_distribution_cache['intensities']
            report.append("Shadow Map Analysis:")
            report.append(f"  Processed Points: {len(shadow_intensities):,}")
            
            if len(shadow_intensities) > 0:
                report.append(f"  Mean Transmission: {np.mean(shadow_intensities):.3f}")
                report.append(f"  Transmission Range: [{np.min(shadow_intensities):.3f}, {np.max(shadow_intensities):.3f}]")
                
                # Gap fraction analysis
                gap_stats = calculate_gap_fraction(self.shadow_distribution_cache, threshold=0.5)
                if gap_stats:
                    report.append("")
                    report.append("Gap Fraction Analysis:")
                    report.append(f"  Gap Fraction: {gap_stats['gap_fraction']:.3f} ({gap_stats['gap_fraction']*100:.1f}%)")
                    report.append(f"  Canopy Cover: {gap_stats['canopy_cover']:.3f} ({gap_stats['canopy_cover']*100:.1f}%)")
                    report.append(f"  Illuminated Points: {gap_stats['gap_points']:,}")
                    report.append(f"  Shadowed Points: {gap_stats['shadowed_points']:,}")
            report.append("")
        
        # Parameters used
        report.append("Analysis Parameters:")
        report.append(f"  Latitude: {self.latitude_input.text()}°")
        report.append(f"  Longitude: {self.longitude_input.text()}°")
        report.append(f"  Date/Time: {self.datetime_input.dateTime().toString('yyyy-MM-dd HH:mm')}")
        
        # Get actual parameters from cache
        if self.voxel_data_cache and len(self.voxel_data_cache) > 2:
            voxel_info = self.voxel_data_cache[2]
            actual_voxel_size = voxel_info.get('voxel_size', 0.05) if voxel_info else 0.05
        else:
            actual_voxel_size = 0.05
        
        if self.ray_data_cache and len(self.ray_data_cache) > 2:
            ray_info = self.ray_data_cache[2]
            actual_ray_spacing = ray_info.get('ray_spacing', 0.05) if ray_info else 0.05
        else:
            actual_ray_spacing = 0.05
        
        report.append(f"  Voxel Size: {actual_voxel_size:.3f} m")
        report.append(f"  Ray Spacing: {actual_ray_spacing:.3f} m")
        report.append("")
        
        # Files saved
        report.append("Output Files:")
        report.append(f"  Directory: {output_dir}/")
        report.append(f"    - analysis_report.txt")
        report.append(f"    - light_intensity_map.png")
        report.append(f"    - shadow_map.png")
        report.append("")
        
        report.append("="*60)
        report.append("END OF REPORT")
        
        # Save everything
        try:
            # Save report
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            self.add_status(f"Report saved: {report_filename}")
            
            # Save light intensity map
            if self.light_distribution_cache:
                self.add_status("Saving light intensity map...")
                self.save_intensity_map(self.light_distribution_cache, light_map_filename, 'light')
            
            # Save shadow map
            if self.shadow_distribution_cache:
                self.add_status("Saving shadow map...")
                self.save_intensity_map(self.shadow_distribution_cache, shadow_map_filename, 'shadow')
            
            self.add_status(f"All files saved to: {output_dir}/")
            
            QMessageBox.information(self, "Export Complete", 
                                f"Analysis completed successfully!\n\n"
                                f"Output directory: {output_dir}/\n\n"
                                f"Files saved:\n"
                                f"- analysis_report.txt\n"
                                f"- light_intensity_map.png\n"
                                f"- shadow_map.png")
        
        except Exception as e:
            self.add_status(f"Error during export: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to complete export: {str(e)}")

    def save_intensity_map(self, distribution, filename, map_type='light'):
        """Save intensity map to file"""
        import matplotlib.pyplot as plt
        from visualization_maps import VisualizationMaps
        
        # Get voxel data
        if self.voxel_data_cache:
            voxel_points, voxel_layers, _ = self.voxel_data_cache
        else:
            voxel_points, voxel_layers = self.points, self.layers
        
        # Get ground points for projection
        ground_mask = (self.layers == 0)
        ground_points = self.points[ground_mask]
        all_points = self.points
        
        # Create projections
        projected_ground = ground_points.copy()
        projected_ground[:, 2] = 0
        
        projected_all = all_points.copy()
        projected_all[:, 2] = 0
        
        # Generate heatmap
        vis_generator = VisualizationMaps(voxel_size=0.05)
        vis_generator.set_ground_points(ground_points)
        
        heatmap_data = vis_generator.generate_heatmap(
            distribution,
            grid_resolution=0.02,
            interpolation_method='cubic',
            smooth_sigma=1.0
        )
        
        if heatmap_data is None:
            self.add_status(f"Warning: Could not generate {map_type} heatmap")
            return
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
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
        
        # Determine colormap and title based on type
        intensities = heatmap_data['intensities']
        mean_intensity = np.mean(intensities)
        
        if map_type == 'light' or mean_intensity > 0.5:
            cmap = 'hot'
            vmin, vmax = 0.4, 1.0
            title = 'Light Intensity Heatmap'
        else:
            cmap = 'gray'
            vmin, vmax = 0.0, 1.0
            title = 'Shadow Distribution Map'
        
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
        
        # Save figure
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        self.add_status(f"Saved: {filename}")
    
    def on_export_error(self, error_msg):
        """Handle export error"""
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
        
        # Re-enable buttons
        for btn in self.buttons.values():
            btn.setEnabled(True)
        
        self.add_status(f"Export error: {error_msg}")
        QMessageBox.critical(self, "Export Error", error_msg)
    
    def add_status(self, message):
        """Add message to status log"""
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.status_text.append(f"[{timestamp}] {message}")
        self.status_text.verticalScrollBar().setValue(
            self.status_text.verticalScrollBar().maximum()
        )


def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
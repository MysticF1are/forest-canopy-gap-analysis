# processor.py - Point Cloud Processing Module

import traceback
from PyQt5.QtCore import QThread, pyqtSignal
from preprocess import preprocess_las


class ProcessorThread(QThread):
    """
    Worker thread for point cloud processing
    Runs preprocessing in background to avoid freezing GUI
    """
    
    # Qt signals for thread communication
    progress_updated = pyqtSignal(str)                      # Progress updates
    processing_finished = pyqtSignal(dict)                  # Success with result dictionary
    processing_error = pyqtSignal(str)                      # Error messages
    
    def __init__(self, filename, cache_dir=None):
        """
        Initialize processor thread
        
        Parameters:
            filename: Path to input file (LAZ/LAS)
            cache_dir: Optional cache directory path
        """
        super().__init__()
        self.filename = filename
        self.cache_dir = cache_dir
        self.is_cancelled = False
    
    def run(self):
        """
        Main processing loop (runs in background thread)
        """
        try:
            # Start processing
            self.progress_updated.emit("Starting preprocessing...")
            
            # Call preprocessing function with progress callback
            result = preprocess_las(
                self.filename,
                self.cache_dir,
                progress_callback=self.on_progress_callback
            )
            
            # Check if result is tuple (legacy) or dict (new format)
            if isinstance(result, tuple):
                # Legacy format: (pcd_path, ply_path, layers_path)
                pcd_path, ply_path, layers_path = result
                result = {
                    'pcd': pcd_path,
                    'ply': ply_path,
                    'layers': layers_path
                }
            
            # Emit success signal
            self.progress_updated.emit("Preprocessing complete!")
            self.processing_finished.emit(result)
            
        except FileNotFoundError as e:
            self.processing_error.emit(f"File not found: {str(e)}")
            
        except ValueError as e:
            self.processing_error.emit(f"Invalid file format: {str(e)}")
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}\n\nDetails:\n{traceback.format_exc()}"
            self.processing_error.emit(error_msg)
    
    def on_progress_callback(self, message):
        """
        Progress callback from preprocessing function
        
        Parameters:
            message: Progress message string
        """
        if not self.is_cancelled:
            self.progress_updated.emit(message)
    
    def stop(self):
        """
        Stop the processing thread
        """
        self.is_cancelled = True
        if self.isRunning():
            self.terminate()
            self.wait()


class ProcessorManager:
    """
    Manager class for handling point cloud processing
    Coordinates between GUI and processing thread
    """
    
    def __init__(self, parent_widget=None):
        """
        Initialize processor manager
        
        Parameters:
            parent_widget: Parent QWidget (optional)
        """
        self.parent_widget = parent_widget
        self.current_thread = None
        self.current_result = None
    
    def start_processing(self, filename, cache_dir=None, 
                        progress_callback=None, 
                        success_callback=None, 
                        error_callback=None):
        """
        Start processing a point cloud file
        
        Parameters:
            filename: Path to input file
            cache_dir: Optional cache directory
            progress_callback: Function to call with progress updates
            success_callback: Function to call on success
            error_callback: Function to call on error
        """
        # Stop any current processing
        self.stop_current_processing()
        
        # Create new thread
        self.current_thread = ProcessorThread(filename, cache_dir)
        
        # Connect signals to callbacks
        if progress_callback:
            self.current_thread.progress_updated.connect(progress_callback)
        
        if success_callback:
            self.current_thread.processing_finished.connect(
                lambda result: self._on_success(result, success_callback)
            )
        
        if error_callback:
            self.current_thread.processing_error.connect(error_callback)
        
        # Start processing
        self.current_thread.start()
    
    def _on_success(self, result, callback):
        """
        Internal success handler
        
        Parameters:
            result: Processing result dictionary
            callback: User callback function
        """
        self.current_result = result
        if callback:
            callback(result)
    
    def stop_current_processing(self):
        """
        Stop current processing if running
        """
        if self.current_thread and self.current_thread.isRunning():
            self.current_thread.stop()
            self.current_thread = None
    
    def is_processing(self):
        """
        Check if currently processing
        
        Returns:
            bool: True if processing is active
        """
        return self.current_thread is not None and self.current_thread.isRunning()
    
    def get_current_result(self):
        """
        Get the result of the last successful processing
        
        Returns:
            dict: Result dictionary or None
        """
        return self.current_result
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.stop_current_processing()
        self.current_result = None
    
    # Convenience methods for main.py compatibility
    def load_file(self, file_path):
        """
        Load and process a file (synchronous wrapper)
        For compatibility with main.py
        """
        # This would need to be implemented based on your needs
        pass
    
    def show_layer(self, layer_id, layer_name):
        """
        Show specific layer visualization
        For compatibility with main.py
        """
        # This would need to be implemented based on your needs
        pass
    
    def generate_heatmap(self, latitude, longitude, datetime):
        """
        Generate light intensity heatmap
        For compatibility with main.py
        """
        # This would need to be implemented based on your needs
        pass
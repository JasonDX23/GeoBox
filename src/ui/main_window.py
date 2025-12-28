## ORIGINAL PySide6 Window with outdated GUI - works completely fine but replaced by sleeker design
# import sys
# import cv2
# import numpy as np
# from PySide6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QSlider, QHBoxLayout
# from PySide6.QtGui import QImage, QPixmap
# from PySide6.QtCore import Qt, Slot, Signal

# from core.kinect import KinectWorker
# from core.processor import TerrainProcessor
# from modules.rain_sim import RainSimulation

# class ARSMainWindow(QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("GeoBox Guru - AR Sandbox")
#         self.processor = TerrainProcessor()
#         self.rain_sim = RainSimulation(count=300)
#         self.contour_interval = 50 # Default starting value

#         # Create Contour Slider
#         self.slider_label = QLabel(f"Contour Interval: {self.contour_interval}")
#         self.interval_slider = QSlider(Qt.Horizontal)
#         self.interval_slider.setMinimum(5)   # Very fine lines
#         self.interval_slider.setMaximum(100) # Very sparse lines
#         self.interval_slider.setValue(self.contour_interval)
#         self.interval_slider.valueChanged.connect(self.update_interval_value)
                
#         self.capture_next_as_base = False
#         self.rain_enabled = False  # Rain starts as disabled

#         # UI Components
#         self.display_label = QLabel("Waiting for Kinect...")
#         self.display_label.setAlignment(Qt.AlignCenter)
        
#         self.calibrate_btn = QPushButton("Capture Base Plane (Reset Sandbox)")
#         self.calibrate_btn.clicked.connect(self.reset_base_plane)
        
#         self.rain_btn = QPushButton("Rain Simulation: OFF")
#         self.rain_btn.clicked.connect(self.toggle_rain)
        
#         # Layout
#         layout = QVBoxLayout()
#         layout.addWidget(self.slider_label)
#         layout.addWidget(self.interval_slider)
#         layout.addWidget(self.display_label)
#         layout.addWidget(self.calibrate_btn)
#         layout.addWidget(self.rain_btn)
        
#         container = QWidget()
#         container.setLayout(layout)
#         self.setCentralWidget(container)

#         # Threading
#         self.worker = KinectWorker(alpha=0.3)
#         self.worker.depth_frame_ready.connect(self.update_frame)
#         self.worker.start()

#     def reset_base_plane(self):
#         """Indented: Now recognized as a method of ARSMainWindow"""
#         self.capture_next_as_base = True
#         print("Calibration triggered...")

#     def toggle_rain(self):
#         """Indented: Toggles the rain logic on/off"""
#         self.rain_enabled = not self.rain_enabled
#         status = "ON" if self.rain_enabled else "OFF"
#         self.rain_btn.setText(f"Rain Simulation: {status}")
        
#     def update_interval_value(self, value):
#         self.contour_interval = value
#         self.slider_label.setText(f"Contour Interval: {self.contour_interval}")

#     @Slot(np.ndarray)
#     def update_frame(self, raw_frame):
#         """Indented: Handles incoming data and rendering"""
#         if self.capture_next_as_base:
#             self.processor.set_base_plane(raw_frame)
#             self.capture_next_as_base = False
#             return 

#         # 1. Processing
#         elevation = self.processor.calculate_elevation(raw_frame)
#         elevation_smooth = cv2.GaussianBlur(elevation, (5,5), 0) # only if needed at short intervals
#         color_terrain = self.processor.apply_color_map(elevation)
        
#         # 2. Contours
#         # quantized = (elevation // 50) * 50
#         # contours = cv2.Canny(quantized.astype(np.uint8), 1, 1)
#         quantized = (elevation_smooth // self.contour_interval) * self.contour_interval
#         contours = cv2.Canny(quantized.astype(np.uint8), 1, 1)
#         color_terrain[contours > 0] = [0, 0, 0]
        
#         # 3. Conditional Rain
#         if self.rain_enabled:
#             dx, dy = self.processor.get_slopes(elevation)
#             self.rain_sim.update(dx, dy)
#             for p in self.rain_sim.particles:
#                 cv2.circle(color_terrain, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
        
#         # 4. Final Render
#         h, w, ch = color_terrain.shape
#         bytes_per_line = ch * w
#         qt_img = QImage(color_terrain.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
#         self.display_label.setPixmap(QPixmap.fromImage(qt_img))

## VERSION 1.2 - sleeker design and dark mode
import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, 
                             QPushButton, QSlider, QHBoxLayout, QFrame, QComboBox)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, Slot, Signal

from core.kinect import KinectWorker
from core.processor import TerrainProcessor, TerrainProcessor_Smoothened
from modules.rain_sim import RainSimulation
from modules.color_maps import ColorMapManager
from modules.contour_match import ContourMatchManager

class ARSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox")
        self.resize(1200, 800)
        self.setStyleSheet("""
        QMainWindow { background-color: #000000; }
        #OverlayPanel { 
            background-color: rgba(30, 30, 30, 180); 
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 30);
        }
        QPushButton { 
            background-color: #2c2c2c; 
            border: 1px solid #444; 
            color: #eee;
            padding: 8px;
        }
        QPushButton:checked {
            background-color: #0078d7; /* Active Blue */
            border-color: #00a4ef;
        }
        QSlider::handle:horizontal {
            background: #00a4ef;
            width: 14px;
            height: 14px;
            margin: -5px 0;
        }
    """)

        self.processor_raw = TerrainProcessor()
        self.processor_filtered = TerrainProcessor_Smoothened()
        self.rain_sim = RainSimulation(count=300)
        self.contour_interval = 50
        self.capture_next_as_base = False
        self.rain_enabled = False
        # For contour matching module
        self.dem_manager = ContourMatchManager()
        self.record_dem_btn = QPushButton("Record Current Sand")
        self.record_dem_btn.clicked.connect(self.record_current_topography)

        self.load_dem_btn = QPushButton("Load DEM")
        self.load_dem_btn.clicked.connect(self.load_target_dem)

        self.active_processor = self.processor_raw
        self.filtering_enabled = False

        # --- UI SETUP ---
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Left Sidebar setup
        sidebar = QWidget()
        sidebar.setObjectName("Sidebar")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(20, 30, 20, 30)

        self.processor_btn = QPushButton('Spatial Filtering: OFF')
        self.processor_btn.setObjectName('FilterBtn')
        self.processor_btn.clicked.connect(self.toggle_filtering)

        title_label = QLabel("Topography Mode")
        title_label.setObjectName("Title")
        
        self.slider_label = QLabel(f"Contour Interval: {self.contour_interval}")
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(5, 100)
        self.interval_slider.setValue(self.contour_interval)
        self.interval_slider.valueChanged.connect(self.update_interval_value)

        self.calibrate_btn = QPushButton("Calibrate Kinect")
        self.calibrate_btn.clicked.connect(self.reset_base_plane)

        self.rain_btn = QPushButton("Rain Simulation: OFF")
        self.rain_btn.setObjectName("RainBtn")
        self.rain_btn.clicked.connect(self.toggle_rain)

        # Right Sidebar Setup
        right_sidebar = QWidget()
        right_sidebar.setObjectName('Sidebar')
        right_sidebar.setFixedWidth(200)

        right_side_layout = QVBoxLayout(right_sidebar)
        right_side_layout.setContentsMargins(20,30,20,30)
        
        # Colour Map Code ----------------
        self.cmap_manager = ColorMapManager()
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.cmap_manager.get_names())
        self.cmap_combo.currentTextChanged.connect(self.cmap_manager.set_map_by_name)
        # ---------------------------------

        # Widget code -------------------
        side_layout.addWidget(title_label)
        side_layout.addWidget(self.slider_label)
        side_layout.addWidget(self.interval_slider)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.calibrate_btn)
        side_layout.addWidget(self.rain_btn)
        side_layout.addWidget(self.processor_btn)
        side_layout.addWidget(QLabel("Colour Map"))
        side_layout.addWidget(self.cmap_combo)
        side_layout.addStretch() # Push everything to the top
        # --------------------------------------------------- #

        # Right side widgets ------------------------
        right_side_layout.addWidget(QLabel("Match Contour Module"))
        right_side_layout.addWidget(self.record_dem_btn)
        right_side_layout.addWidget(self.load_dem_btn)
        right_side_layout.addStretch()

        # Viewport setup
        self.display_label = QLabel("Initializing Depth Stream...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000;")

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.display_label, 1) # 1 makes it expand
        main_layout.addWidget(right_sidebar)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.worker = KinectWorker(alpha=0.3)
        self.worker.depth_frame_ready.connect(self.update_frame)
        self.worker.start()

    def reset_base_plane(self):
        self.capture_next_as_base = True

    def toggle_filtering(self):
        'Toggles between raw and smoothened processing logic'
        self.filtering_enabled = not self.filtering_enabled
        
        if self.filtering_enabled:
            self.active_processor = self.processor_filtered
            self.processor_btn.setText("Spatial Filtering: ON")
            # Sync the base plane so the transition is seamless
            if self.processor_raw.base_plane is not None:
                self.processor_filtered.set_base_plane(self.processor_raw.base_plane)
        else:
            self.active_processor = self.processor_raw
            self.processor_btn.setText("Spatial Filtering: OFF")

    def toggle_rain(self):
        self.rain_enabled = not self.rain_enabled
        status = "ON" if self.rain_enabled else "OFF"
        self.rain_btn.setText(f"Rain Simulation: {status}")
        self.rain_btn.setProperty("active", str(self.rain_enabled).lower())
        self.rain_btn.style().unpolish(self.rain_btn) # Force style refresh
        self.rain_btn.style().polish(self.rain_btn)

    def update_interval_value(self, value):
        self.contour_interval = value
        self.slider_label.setText(f"Contour Interval: {self.contour_interval}")

    def record_current_topography(self):
    # Capture the current live frame and save it
    # raw_frame here would be the latest from your worker
        if hasattr(self, 'last_raw_frame'):
            self.dem_manager.save_current_sand_as_dem(self.last_raw_frame)

    def load_target_dem(self):
        # Use a file dialog to pick a .dem (NumPy) file
        from PySide6.QtWidgets import QFileDialog
        fname, _ = QFileDialog.getOpenFileName(self, "Open DEM", "", "DEM Files (*.dem.npy)")
        if fname:
            self.dem_manager.load_dem(fname)

    @Slot(np.ndarray)
    def update_frame(self, raw_frame):
        # Store frame for the record_current_topography method
        self.last_raw_frame = raw_frame.copy()

        # 1. Base Plane Capture
        if self.capture_next_as_base:
            self.active_processor.set_base_plane(raw_frame)
            self.capture_next_as_base = False
            return 

        # 2. Elevation and Coloring
        elevation = self.active_processor.calculate_elevation(raw_frame)
        elevation_smooth = cv2.GaussianBlur(elevation, (5,5), 0)

        if self.dem_manager.is_matching_mode:
            color_terrain, score = self.dem_manager.calculate_matching_guide(raw_frame)
            cv2.putText(color_terrain, f"RMSE: {score:.2f}mm", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            #elevation_smooth = cv2.GaussianBlur(elevation, (5,5), 0)
            norm_elevation = np.clip(elevation_smooth, 0,255).astype(np.uint8)
            color_terrain = self.cmap_manager.apply(norm_elevation)
        
        # 3. Contours
        quantized = (elevation_smooth // self.contour_interval) * self.contour_interval
        contours = cv2.Canny(quantized.astype(np.uint8), 1, 1)
        color_terrain[contours > 0] = [0, 0, 0]
        
        # 4. Rain Simulation (FIXED VARIABLE NAME HERE)
        if self.rain_enabled:
            # We use active_processor so rain flows correctly on filtered sand
            dx, dy = self.active_processor.get_slopes(elevation) 
            self.rain_sim.update(dx, dy)
            for p in self.rain_sim.particles:
                cv2.circle(color_terrain, (int(p[0]), int(p[1])), 2, (255, 0, 0), -1)
        
        # 5. Render to UI
        h, w, ch = color_terrain.shape
        qt_img = QImage(color_terrain.data.tobytes(), w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.display_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
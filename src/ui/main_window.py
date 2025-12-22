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
                             QPushButton, QSlider, QHBoxLayout, QFrame)
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt, Slot, Signal

from core.kinect import KinectWorker
from core.processor import TerrainProcessor, TerrainProcessor_Smoothened
from modules.rain_sim import RainSimulation

class ARSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox")
        self.resize(1200, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget#Sidebar { 
                background-color: #1e1e1e; 
                border-right: 1px solid #333;
                min-width: 280px;
            }
            QLabel { color: #e0e0e0; font-family: 'Segoe UI'; font-size: 13px; }
            QLabel#Title { font-size: 18px; font-weight: bold; color: #3d5afe; margin-bottom: 10px; }
            QPushButton { 
                background-color: #3d5afe; color: white; border-radius: 4px; 
                padding: 12px; font-weight: bold; border: none; margin-top: 5px;
            }
            QPushButton:hover { background-color: #536dfe; }
            QPushButton#RainBtn[active="true"] { background-color: #f44336; }
            QSlider::groove:horizontal { height: 4px; background: #333; border-radius: 2px; }
            QSlider::handle:horizontal { 
                background: #3d5afe; width: 16px; height: 16px; 
                margin: -6px 0; border-radius: 8px; 
            }
        """)

        self.processor_raw = TerrainProcessor()
        self.processor_filtered = TerrainProcessor_Smoothened()
        self.rain_sim = RainSimulation(count=300)
        self.contour_interval = 50
        self.capture_next_as_base = False
        self.rain_enabled = False

        self.active_processor = self.processor_raw
        self.filtering_enabled = False

        # --- UI SETUP ---
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar setup
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

        side_layout.addWidget(title_label)
        side_layout.addWidget(self.slider_label)
        side_layout.addWidget(self.interval_slider)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.calibrate_btn)
        side_layout.addWidget(self.rain_btn)
        side_layout.addWidget(self.processor_btn)
        side_layout.addStretch() # Push everything to the top

        # Viewport setup
        self.display_label = QLabel("Initializing Depth Stream...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000;")

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.display_label, 1) # 1 makes it expand

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

    @Slot(np.ndarray)
    def update_frame(self, raw_frame):
        # 1. Base Plane Capture
        if self.capture_next_as_base:
            self.active_processor.set_base_plane(raw_frame)
            self.capture_next_as_base = False
            return 

        # 2. Elevation and Coloring
        elevation = self.active_processor.calculate_elevation(raw_frame)
        elevation_smooth = cv2.GaussianBlur(elevation, (5,5), 0)
        color_terrain = self.active_processor.apply_color_map(elevation)
        
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
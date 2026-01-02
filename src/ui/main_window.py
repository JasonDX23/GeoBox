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
import os
from PySide6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar,
                             QPushButton, QSlider, QHBoxLayout, QFrame, QComboBox, QApplication)
from PySide6.QtGui import QImage, QPixmap, QFont, QPen, QPainter
from PySide6.QtCore import Qt, Slot, Signal, QRect

from core.kinect import KinectWorker
from core.processor import TerrainProcessor, TerrainProcessor_Smoothened
from modules.color_maps import ColorMapManager
from modules.contour_match import ContourMatchManager
from core.KinectProjector_calibration import KinectProjector

class ProjectorWindow(QWidget):
    """The dedicated full-screen window for the projector (Secondary Screen)."""
    def __init__(self, screen_index=1):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        screens = QApplication.screens()
        # Fallback to primary if secondary is not detected
        target_screen = screens[screen_index] if len(screens) > screen_index else screens[0]
        self.setGeometry(target_screen.geometry())
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.label)
        self.showFullScreen()

    def display_pattern(self, pattern_img):
        pattern_img = np.ascontiguousarray(pattern_img)
        h, w, ch = pattern_img.shape
        q_img = QImage(pattern_img.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_img))
        self.label.repaint()

class ARSMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox AR Sandbox")
        self.resize(1200, 800)
        
        # --- Core Logic Components ---
        self.calibrator = KinectProjector(1024, 768)
        self.projector_ui = ProjectorWindow(screen_index=1)
        self.processor_raw = TerrainProcessor()
        self.processor_filtered = TerrainProcessor_Smoothened()
        self.active_processor = self.processor_raw
        
        self.cmap_manager = ColorMapManager()
        self.dem_manager = ContourMatchManager()
        
        # --- State Variables ---
        self.contour_interval = 20
        self.capture_next_as_base = False
        self.filtering_enabled = False
        self.is_calibrating_roi = False

        # --- UI Initialization ---
        self.init_ui()
        
        # --- Data Stream ---
        self.worker = KinectWorker(alpha=0.3)
        self.worker.depth_frame_ready.connect(self.update_frame)
        self.worker.start()

        # --- Load Calibration Matrix ---
        self.load_calibration()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar Setup
        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        side_layout = QVBoxLayout(sidebar)

        # UI Controls
        self.slider_label = QLabel(f"Contour Interval: {self.contour_interval}")
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(5, 100)
        self.interval_slider.setValue(self.contour_interval)
        self.interval_slider.valueChanged.connect(self.update_interval_value)

        self.calibrate_btn = QPushButton("Calibrate Kinect (Base)")
        self.calibrate_btn.clicked.connect(self.reset_base_plane)

        self.processor_btn = QPushButton('Spatial Filtering: OFF')
        self.processor_btn.clicked.connect(self.toggle_filtering)

        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(self.cmap_manager.get_names())
        self.cmap_combo.currentTextChanged.connect(self.cmap_manager.set_map_by_name)

        self.roi_btn = QPushButton("Set ROI Boundary")
        self.roi_btn.clicked.connect(self.enter_roi_mode)

        # Assemble Sidebar
        side_layout.addWidget(QLabel("GEOBOX CONTROLS"))
        side_layout.addWidget(self.slider_label)
        side_layout.addWidget(self.interval_slider)
        side_layout.addSpacing(20)
        side_layout.addWidget(self.calibrate_btn)
        side_layout.addWidget(self.processor_btn)
        side_layout.addWidget(QLabel("Color Map Selection"))
        side_layout.addWidget(self.cmap_combo)
        side_layout.addWidget(self.roi_btn)
        side_layout.addStretch()

        # Main Display
        self.display_label = QLabel("Initializing...")
        self.display_label.setAlignment(Qt.AlignCenter)
        self.display_label.setStyleSheet("background-color: #000;")
        
        self.roi_selector = ROISelectorLabel()
        self.roi_selector.roi_selected.connect(self.finalize_roi)
        self.roi_selector.hide()

        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.display_label, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_calibration(self):
        """Attempts to load the 3x3 Homography Matrix from the utility."""
        file_path = "homography_matrix.npy"
        if os.path.exists(file_path):
            self.calibrator.homography = np.load(file_path)
            self.calibrator._is_calibrated = True
            print("Successfully loaded Projector Homography.")
        else:
            print("No Calibration File Found. Projector output will not be warped.")

    def reset_base_plane(self):
        """Triggered by the 'Calibrate Kinect' button to set the sand floor."""
        self.capture_next_as_base = True
        print("System ready. Capturing next frame as base plane...")

    def start_calibration(self):
        """Fallback for the 'Calibrate Projector' button if needed in-app."""
        # Since we moved to a standalone app, this could just show a reminder
        print("Please use the standalone Calibration Utility for projector alignment.")

    @Slot(np.ndarray)
    def update_frame(self, raw_frame):
        self.last_raw_frame = raw_frame.copy()

        # 1. Handle Base Plane Calibration
        if self.capture_next_as_base:
            if self.active_processor.roi is None:
                self.active_processor.update_roi(0, 0, 640, 480)
            self.active_processor.set_base_plane(raw_frame)
            self.capture_next_as_base = False
            return

        # 2. Elevation Processing
        elevation = self.active_processor.calculate_elevation(raw_frame)
        elevation_smooth = cv2.GaussianBlur(elevation, (5, 5), 0)

        # 3. Coloring and Contours
        norm_for_lut = np.clip(((elevation_smooth + 250) / 500) * 255, 0, 255).astype(np.uint8)
        color_terrain = self.cmap_manager.apply(norm_for_lut)

        quantized = (elevation_smooth // self.contour_interval) * self.contour_interval
        contours = cv2.Canny(quantized.astype(np.uint8), 1, 1)
        color_terrain[contours > 0] = [0, 0, 0]

        # 4. Render to Main GUI (Laptop screen)
        h, w, ch = color_terrain.shape
        qt_img = QImage(color_terrain.data.tobytes(), w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.display_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.display_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 5. Render to Projector (Warped Perspective)
        if self.calibrator._is_calibrated:
            # Magic-Sand often uses a 1.05x or 1.1x scale factor 
            # to "over-fill" the sandbox and hide black borders
            scale_factor = 1.05
            M = self.calibrator.homography.copy()
            M[0,0] *= scale_factor
            M[1,1] *= scale_factor
            
            warped = cv2.warpPerspective(
                color_terrain, 
                M, 
                (1024, 768),
                flags=cv2.INTER_LINEAR
            )
            self.projector_ui.display_pattern(warped)

    def toggle_filtering(self):
        self.filtering_enabled = not self.filtering_enabled
        self.active_processor = self.processor_filtered if self.filtering_enabled else self.processor_raw
        self.processor_btn.setText(f"Spatial Filtering: {'ON' if self.filtering_enabled else 'OFF'}")

    def update_interval_value(self, value):
        self.contour_interval = value
        self.slider_label.setText(f"Contour Interval: {self.contour_interval}")

    def enter_roi_mode(self):
        if hasattr(self, 'last_raw_frame'):
            visible_frame = cv2.normalize(self.last_raw_frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w = visible_frame.shape
            qimg = QImage(visible_frame.data, w, h, w, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimg).scaled(self.display_label.size(), Qt.KeepAspectRatio)
            self.roi_selector.setPixmap(pixmap)
            self.display_label.hide()
            self.centralWidget().layout().replaceWidget(self.display_label, self.roi_selector)
            self.roi_selector.show()

    @Slot(int, int, int, int)
    def finalize_roi(self, x, y, w, h):
        """Maps UI coordinates to Kinect 640x480 sensor space."""
        label_w = self.roi_selector.width()
        label_h = self.roi_selector.height()
        
        # Calculate scaling factor and offsets (aspect ratio 4:3)
        scale = min(label_w / 640, label_h / 480)
        offset_x = (label_w - (640 * scale)) / 2
        offset_y = (label_h - (480 * scale)) / 2

        # Map and Clamp to 640x480 boundaries
        sensor_x = int(max(0, min((x - offset_x) / scale, 639)))
        sensor_y = int(max(0, min((y - offset_y) / scale, 479)))
        sensor_w = int(min(w / scale, 640 - sensor_x))
        sensor_h = int(min(h / scale, 480 - sensor_y))

        self.active_processor.update_roi(sensor_x, sensor_y, sensor_w, sensor_h)
        
        # Reset UI view
        self.roi_selector.hide()
        self.centralWidget().layout().replaceWidget(self.roi_selector, self.display_label)
        self.display_label.show()
        self.capture_next_as_base = True # Recalibrate base for new ROI

class ROISelectorLabel(QLabel):
    roi_selected = Signal(int, int, int, int)
    def __init__(self):
        super().__init__()
        self.selecting = False
        self.start_p = None

    def mousePressEvent(self, event):
        self.start_p = event.position().toPoint()
        self.selecting = True

    def mouseReleaseEvent(self, event):
        if self.selecting:
            end_p = event.position().toPoint()
            self.roi_selected.emit(self.start_p.x(), self.start_p.y(), 
                                 abs(end_p.x()-self.start_p.x()), abs(end_p.y()-self.start_p.y()))
            self.selecting = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selecting and self.start_p:
            curr_p = self.mapFromGlobal(self.cursor().pos())
            painter = QPainter(self)
            painter.setPen(QPen(Qt.red, 2, Qt.DashLine))
            rect = QRect(self.start_p, curr_p)
            painter.drawRect(rect)
            self.update() # Keep the line following the mouse
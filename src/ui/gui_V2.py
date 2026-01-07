import numpy as np
import cv2
import sys
import json
import os
from PySide6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar,
                             QPushButton, QSlider, QHBoxLayout, QFrame, QComboBox, QApplication)
from PySide6.QtGui import QImage, QPixmap, QFont, QPen, QPainter
from PySide6.QtCore import Qt, Slot, Signal, QRect

from core.kinect import KinectWorker
from core.processor import TerrainProcessor
from core.KinectProjector import KinectProjector

# class ProjectorWindow(QWidget):
#     """The dedicated full-screen window for the projector (Secondary Screen)."""
#     def __init__(self, screen_index=1):
#         super().__init__()
#         self.setWindowFlags(Qt.FramelessWindowHint)
#         screens = QApplication.screens()
#         # Fallback to primary if secondary is not detected
#         target_screen = screens[screen_index] if len(screens) > screen_index else screens[0]
#         self.setGeometry(target_screen.geometry())
        
#         self.label = QLabel(self)
#         self.label.setAlignment(Qt.AlignCenter)
#         self.label.setStyleSheet("background-color: black;")
#         self.layout = QVBoxLayout(self)
#         self.layout.setContentsMargins(0, 0, 0, 0)
#         self.layout.addWidget(self.label)
#         self.showFullScreen()

#     def display_pattern(self, pattern_img):
#         pattern_img = np.ascontiguousarray(pattern_img)
#         h, w, ch = pattern_img.shape
#         q_img = QImage(pattern_img.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
#         self.label.setPixmap(QPixmap.fromImage(q_img))
#         self.label.repaint()

class GeoBox(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GeoBox')
        self.resize(1200, 800)
        
        # Core Logic
        self.worker = KinectWorker()
        self.processor = TerrainProcessor()
        self.calibration = KinectProjector(1024, 768)
        #self.projector_ui = ProjectorWindow(screen_index=1)
        
        # State Variables
        self.contour_interval = 20
        self.capture_next_as_base = False
        self.filtering_enabled = False
        self.is_calibrating_roi = False

        self.init_ui()

        # Start the Kinect
        self.worker.depth_frame_ready.connect(self.update_frame)
        self.worker.start()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(250)
        side_layout = QVBoxLayout(sidebar)

        # UI Controls
        self.slider_label = QLabel(f"Contour Interval: {self.contour_interval}")
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(5,100)
        self.interval_slider.setValue((self.contour_interval))
        self.interval_slider.valueChanged.connect(self.update_interval_value)
        

        side_layout.addWidget(QLabel("GeoBox Controls"))
        side_layout.addWidget(self.slider_label)
        side_layout.addWidget(self.interval_slider)
        side_layout.addSpacing(20)
        # add sidelayout calibration and roi buttons
        side_layout.addStretch()

        self.display_label = QLabel("Initializing...")
        self.display_label.setAlignment(Qt.AlignCenter)
        
        main_layout.addWidget(sidebar)
        main_layout.addWidget(self.display_label, 1)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    # modules
    def load_calibration(self, filename="calibration.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.projection_matrix = np.array(data["projection_matrix"])
                self.camera_matrix = np.array(data["camera_matrix"])
            return True
        except:
            return False
        
    def

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


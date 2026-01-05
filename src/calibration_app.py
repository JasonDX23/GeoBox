import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QPushButton, QProgressBar, QWidget)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap

from core.kinect import KinectWorker
from core.KinectProjector import KinectProjector
from core.processor import TerrainProcessor

class ProjectorWindow(QWidget):
    """Handles the secondary display on the projector."""
    def __init__(self, screen_index=1):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint)
        screens = QApplication.screens()
        # Fallback to primary if secondary isn't found
        target_screen = screens[screen_index] if len(screens) > screen_index else screens[0]
        self.setGeometry(target_screen.geometry())
        
        self.label = QLabel(self)
        self.label.setScaledContents(False)
        self.label.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.showFullScreen()

    def display_pattern(self, pattern):
        h, w, ch = pattern.shape
        # Use RGB888 and rgbSwapped for correct color rendering
        q_img = QImage(pattern.data, w, h, ch * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(q_img))

class CalibrationUtility(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox Pro Calibration")
        
        # Core Components
        self.projector_res = (1024, 768)
        self.calibrator = KinectProjector(*self.projector_res)
        self.processor = TerrainProcessor() # Used to help visualize depth
        
        # State Management
        self.positions = [
            (50, 50), (450, 50), (850, 50),
            (50, 350), (450, 350), (850, 350),
            (50, 650), (450, 650), (850, 650)
        ]
        self.current_pos_idx = 0
        self.is_raised_round = False
        self.captured_at_current_pos = False

        # UI Setup
        self.init_ui()
        
        # Hardware Connection
        self.projector = ProjectorWindow(screen_index=1)
        self.worker = KinectWorker()
        self.worker.depth_frame_ready.connect(self.on_depth_ready)
        self.worker.start()

        self.last_depth = None
        self.update_pattern()
        
        # Processing Timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_view)
        self.timer.start(33)

    def init_ui(self):
        self.label_status = QLabel("Ready to start...")
        self.view_rgb = QLabel("Kinect RGB")
        self.btn_capture = QPushButton("CAPTURE POSITION")
        self.btn_capture.clicked.connect(self.capture_current_frame)
        
        layout = QVBoxLayout()
        layout.addWidget(self.label_status)
        layout.addWidget(self.view_rgb)
        layout.addWidget(self.btn_capture)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def on_depth_ready(self, depth_frame):
        self.last_depth = depth_frame

    def update_pattern(self):
        x, y = self.positions[self.current_pos_idx]
        # Draw 7x6 chessboard pattern
        pattern = self.calibrator.generate_pattern(x, y, size=300)
        self.projector.display_pattern(pattern)
        self.label_status.setText(f"Position {self.current_pos_idx+1}/{len(self.positions)}")

    def capture_current_frame(self):
        """Captures a 3D snapshot: RGB pixels + Depth mm"""
        rgb = self.worker.get_latest_rgb()
        depth = self.last_depth
        
        if rgb is None or depth is None:
            return

        # Find 2D corners in RGB
        ret, corners = cv2.findChessboardCorners(rgb, (6, 5))
        if ret:
            # 1. Store the Projector's 2D target points
            proj_pts = self.calibrator.get_target_pts_for_pos(self.current_pos_idx)
            
            # 2. Store the Kinect's 3D points (X, Y, Depth)
            for i, corner in enumerate(corners):
                cx, cy = corner.ravel()
                # Crucial: Grab the depth at the corner coordinate
                z = depth[int(cy), int(cx)]
                if z > 0:
                    self.calibrator.add_point_pair(proj_pts[i], (cx, cy, z))
            
            self.advance_state()
        else:
            self.label_status.setText("FAILED: Could not find chessboard. Adjust sand.")

    def advance_state(self):
        self.current_pos_idx += 1
        if self.current_pos_idx >= len(self.positions):
            if not self.is_raised_round:
                self.is_raised_round = True
                self.current_pos_idx = 0
                self.label_status.setText("LEVEL 1 DONE. Place board on sand.")
            else:
                self.finalize()
        self.update_pattern()

    def finalize(self):
        self.calibrator.solve_matrix()
        self.calibrator.save_to_file("calibration_data.json")
        self.label_status.setText("CALIBRATION SAVED. You can close now.")
        self.timer.stop()

    def update_view(self):
        rgb = self.worker.get_latest_rgb()
        if rgb is not None:
            # Display live preview with a status overlay
            h, w, ch = rgb.shape
            qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).rgbSwapped()
            self.view_rgb.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationUtility()
    window.show()
    sys.exit(app.exec())
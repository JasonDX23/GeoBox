import sys
import cv2
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QPushButton, QProgressBar, QWidget)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QImage, QPixmap

from core.kinect import KinectWorker
from core.KinectProjector_calibration import KinectProjector

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
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("background-color: black;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.label)
        self.showFullScreen()

    def display_pattern(self, pattern):
        h, w, ch = pattern.shape
        # Use RGB888 and rgbSwapped for correct color rendering
        q_img = QImage(pattern.data, w, h, ch * w, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(q_img))

class CalibrationUtility(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox Multi-Level Calibration")
        self.resize(800, 600)
        self.calibrator = KinectProjector(1024, 768)
        
        # Calibration States: (X, Y, Name)
        self.positions = [
        (50, 50, "Top-Left"),      (272, 50, "Top-Center"),    (494, 50, "Top-Right"),
        (50, 184, "Mid-Left"),     (272, 184, "Center"),       (494, 184, "Mid-Right"),
        (50, 318, "Bot-Left"),     (272, 318, "Bot-Center"),   (494, 318, "Bot-Right")
    ]
        self.current_pos_idx = 0
        self.is_raised_round = False
        
        # UI
        self.label = QLabel("Initializing...")
        self.progress = QProgressBar()
        self.progress.setRange(0, 84) # 42 corners * 2 frames
        self.btn_next = QPushButton("Start Next Position")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self.advance_state)
        
        self.view = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.view)
        layout.addWidget(self.progress)
        layout.addWidget(self.btn_next)
        
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.projector = ProjectorWindow(screen_index=1)
        self.worker = KinectWorker()
        self.worker.start()
        
        self.update_pattern()
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_loop)
        self.timer.start(100)

    def update_pattern(self):
        off_x, off_y, name = self.positions[self.current_pos_idx]
        round_name = "RAISED BOARD" if self.is_raised_round else "FLAT SAND"
        self.label.setText(f"CURRENT ROUND: {round_name}\nTarget: {name}")
        
        # 1. Draw the visual pattern for the projector
        pattern = np.full((768, 1024, 3), 255, dtype=np.uint8)
        sq = 80
        for y in range(5):
            for x in range(6):
                if (x + y) % 2 == 0:
                    cv2.rectangle(pattern, (off_x + x*sq, off_y + y*sq), 
                                 (off_x + (x+1)*sq, off_y + (y+1)*sq), (0,0,0), -1)
        self.projector.display_pattern(pattern)
        
        # 2. FIX: Get the target coordinates from the calibrator class
        # This replaces the manual loop and the .append() call
        self.current_targets = self.calibrator.generate_projector_points(off_x, off_y, sq_sz=sq)
        
        self.temp_kinect_pts = []
        self.progress.setValue(0)

    def process_loop(self):
        rgb = self.worker.get_latest_rgb()
        if rgb is None or self.btn_next.isEnabled(): return
        
        corners = self.calibrator.find_chessboard(rgb)
        
        # Only proceed if detection is successful
        if corners is not None:
            # We only want to save the points IF we haven't finished this position yet
            if len(self.temp_kinect_pts) < 84: # (42 corners * 2 frames)
                self.temp_kinect_pts.extend(corners.reshape(-1, 2))
                
                # CRITICAL: Append the CURRENT targets for every set of corners found
                self.calibrator.collected_kinect_points.extend(corners.reshape(-1, 2))
                self.calibrator.collected_projector_points.extend(self.current_targets)
                
                self.progress.setValue(len(self.temp_kinect_pts))
                
            if len(self.temp_kinect_pts) >= 84:
                self.btn_next.setEnabled(True)
                self.label.setText("Position Captured! Move sand/board and click Next.")
        # Feedback view
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888).rgbSwapped()
        self.view.setPixmap(QPixmap.fromImage(qimg).scaled(640, 480, Qt.KeepAspectRatio))

    def advance_state(self):
        self.current_pos_idx += 1
        if self.current_pos_idx >= len(self.positions):
            if not self.is_raised_round:
                self.is_raised_round = True
                self.current_pos_idx = 0
                self.label.setText("SAND LEVEL DONE. Place board on top and click 'Start Next Position'")
                self.btn_next.setEnabled(True)
                return
            else:
                self.finalize_calibration()
                return
        self.btn_next.setEnabled(False)
        self.update_pattern()

    def finalize_calibration(self):
        if self.calibrator.calculate_mapping():
            # Save to the shared UI folder
            np.save("homography_matrix.npy", self.calibrator.homography)
            self.label.setText("CALIBRATION SUCCESSFUL! matrix saved to ui/homography_matrix.npy")
            self.timer.stop()
            self.projector.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CalibrationUtility()
    window.show()
    sys.exit(app.exec())
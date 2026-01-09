import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QTabWidget, QSizePolicy, QRubberBand  # <--- Moved here
)
from PySide6.QtCore import Qt, Signal, Slot, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPainter # <--- Removed QRubberBand from here

# --- Internal Helper Widgets ---

class ROISelector(QWidget):
    """Transparent overlay for drawing the calibration rectangle"""
    roi_selected = Signal(list) # Emits [x, y, w, h]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.is_selecting = False
        self.setStyleSheet("background-color: transparent;")
        
    def mousePressEvent(self, event):
        self.origin = event.pos()
        self.rubberBand.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand.show()
        self.is_selecting = True

    def mouseMoveEvent(self, event):
        if self.is_selecting:
            self.rubberBand.setGeometry(QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event):
        self.is_selecting = False
        rect = self.rubberBand.geometry()
        self.roi_selected.emit([rect.x(), rect.y(), rect.width(), rect.height()])

class VideoWidget(QWidget):
    """Renders video frame without forcing layout expansion"""
    def __init__(self):
        super().__init__()
        self.image = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: #111; border: 1px solid #444;")

    def set_frame(self, qt_img):
        self.image = qt_img
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.image:
            w_widget = self.width()
            h_widget = self.height()
            scaled = self.image.scaled(w_widget, h_widget, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (w_widget - scaled.width()) // 2
            y = (h_widget - scaled.height()) // 2
            painter.drawImage(x, y, scaled)
        else:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Awaiting Stream...")

# --- Main Dashboard Class ---

class GeoBoxDashboard(QMainWindow):
    class Signals(QWidget):
        # Visuals
        sea_changed = Signal(float)
        mode_preset = Signal(str)
        mode_custom = Signal(list)
        contours_toggled = Signal(bool)
        contour_interval = Signal(float)
        # Fluids
        rain_changed = Signal(bool, float)
        evap_changed = Signal(float)
        clear_water = Signal()
        # Calibration
        start_roi_select = Signal()
        save_calibration = Signal()
        capture_base = Signal() # New
        capture_top = Signal()  # New

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox Studio - Integrated")
        self.resize(1100, 768)
        self.setStyleSheet("QMainWindow { background-color: #333; color: #ccc; }")
        
        self.signals = self.Signals()
        
        # Central Container
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # ================= Sidebar =================
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout(sidebar)
        
        # 1. Global Settings
        gb_global = QGroupBox("Global Geometry")
        l_global = QVBoxLayout()
        self.sea_slider = QSlider(Qt.Horizontal)
        self.sea_slider.setRange(-50, 50)
        self.sea_slider.valueChanged.connect(lambda v: self.signals.sea_changed.emit(v/100.0))
        l_global.addWidget(QLabel("Sea Level Offset"))
        l_global.addWidget(self.sea_slider)
        gb_global.setLayout(l_global)
        
        # 2. Tabs Initialization
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #555; }")

        # --- Tab A: Visuals ---
        t_vis = QWidget()
        l_vis = QVBoxLayout()
        
        self.combo = QComboBox()
        self.combo.addItems(['terrain', 'ocean', 'gist_earth', 'viridis', 'magma'])
        self.combo.currentTextChanged.connect(self.signals.mode_preset.emit)
        l_vis.addWidget(QLabel("Color Theme:"))
        l_vis.addWidget(self.combo)
        
        l_vis.addWidget(QLabel("\nTopographic Lines:"))
        self.btn_cont = QPushButton("Toggle Contours")
        self.btn_cont.setCheckable(True)
        self.btn_cont.toggled.connect(self.signals.contours_toggled.emit)
        l_vis.addWidget(self.btn_cont)
        
        self.slider_cont = QSlider(Qt.Horizontal)
        self.slider_cont.setRange(2, 20)
        self.slider_cont.setValue(5)
        self.slider_cont.valueChanged.connect(lambda v: self.signals.contour_interval.emit(v/100.0))
        l_vis.addWidget(self.slider_cont)
        l_vis.addStretch()
        t_vis.setLayout(l_vis)
        
        # --- Tab B: Fluids ---
        t_fluid = QWidget()
        l_fluid = QVBoxLayout()
        
        self.chk_rain = QCheckBox("Enable Rainfall")
        self.chk_rain.toggled.connect(self._update_rain)
        l_fluid.addWidget(self.chk_rain)
        
        l_fluid.addWidget(QLabel("Rain Intensity:"))
        self.sld_rain = QSlider(Qt.Horizontal)
        self.sld_rain.setRange(1, 50)
        self.sld_rain.setValue(10)
        self.sld_rain.valueChanged.connect(self._update_rain)
        l_fluid.addWidget(self.sld_rain)
        
        l_fluid.addWidget(QLabel("Evaporation Rate:"))
        self.sld_evap = QSlider(Qt.Horizontal)
        self.sld_evap.setRange(0, 50)
        self.sld_evap.setValue(1)
        self.sld_evap.valueChanged.connect(lambda v: self.signals.evap_changed.emit(v/1000.0))
        l_fluid.addWidget(self.sld_evap)
        
        btn_clear = QPushButton("Drain System")
        btn_clear.clicked.connect(self.signals.clear_water.emit)
        l_fluid.addWidget(btn_clear)
        l_fluid.addStretch()
        t_fluid.setLayout(l_fluid)

        # --- Tab C: Calibration ---
        t_calib = QWidget()
        l_calib = QVBoxLayout()
        
        l_calib.addWidget(QLabel("<b>Step 1: ROI</b>"))
        btn_roi = QPushButton("Draw Sand Box")
        btn_roi.clicked.connect(self.signals.start_roi_select.emit)
        l_calib.addWidget(btn_roi)
        
        l_calib.addWidget(QLabel("<b>Step 2: Base Level</b>"))
        l_calib.addWidget(QLabel("Remove all sand or flatten it."))
        btn_base = QPushButton("Capture Base Pattern")
        btn_base.clicked.connect(self.signals.capture_base.emit)
        l_calib.addWidget(btn_base)

        l_calib.addWidget(QLabel("<b>Step 3: Top Level</b>"))
        l_calib.addWidget(QLabel("Place White Board on top."))
        btn_top = QPushButton("Capture Top Pattern")
        btn_top.clicked.connect(self.signals.capture_top.emit)
        l_calib.addWidget(btn_top)
        
        l_calib.addWidget(QLabel("<b>Step 4: Finalize</b>"))
        btn_save = QPushButton("Compute & Save")
        btn_save.clicked.connect(self.signals.save_calibration.emit)
        l_calib.addWidget(btn_save)
        
        l_calib.addStretch()
        t_calib.setLayout(l_calib)

        # Add Tabs
        self.tabs.addTab(t_vis, "Visuals")
        self.tabs.addTab(t_fluid, "Fluids")
        self.tabs.addTab(t_calib, "Calibration")

        # Assemble Sidebar
        side_layout.addWidget(gb_global)
        side_layout.addWidget(self.tabs)
        side_layout.addStretch()
        
        # ================= Preview =================
        self.video = VideoWidget()
        
        # ROI Overlay (Parented to video)
        self.roi_overlay = ROISelector(self.video)
        self.roi_overlay.hide()
        self.roi_overlay.roi_selected.connect(self._on_roi_drawn)
        
        layout.addWidget(sidebar)
        layout.addWidget(self.video, stretch=1)

    # --- Slots & Handlers ---

    def _update_rain(self):
        enabled = self.chk_rain.isChecked()
        rate = self.sld_rain.value() / 1000.0
        self.signals.rain_changed.emit(enabled, rate)

    def _on_roi_drawn(self, rect_list):
        self.roi_overlay.hide()
        # Scale logic for ROI (UI to Engine coords)
        w_wid = self.video.width()
        h_wid = self.video.height()
        if w_wid == 0 or h_wid == 0: return

        scale_x = 640.0 / w_wid
        scale_y = 480.0 / h_wid
        
        final_roi = [
            int(rect_list[0] * scale_x),
            int(rect_list[1] * scale_y),
            int(rect_list[2] * scale_x),
            int(rect_list[3] * scale_y)
        ]
        
        print(f"ROI Captured (UI->Engine): {final_roi}")
        # Re-emit for Engine
        self.roi_overlay.roi_selected.emit(final_roi)

    @Slot()
    def toggle_roi_mode(self):
        self.roi_overlay.resize(self.video.size())
        self.roi_overlay.show()

    @Slot(np.ndarray)
    def update_feed(self, frame):
        # Handle Gray (2D) vs Color (3D)
        if len(frame.shape) == 2:
            h, w = frame.shape
            ch = 1
            # For 2D Grayscale, stride is just width
            bytes_per_line = w
            fmt = QImage.Format_Grayscale8
        else:
            h, w, ch = frame.shape
            bytes_per_line = ch * w
            fmt = QImage.Format_BGR888

        # Create QImage with correct parameters
        img = QImage(frame.data, w, h, bytes_per_line, fmt)
        self.video.set_frame(img.copy())
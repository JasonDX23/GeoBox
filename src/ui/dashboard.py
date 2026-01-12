import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QSlider, QComboBox, QCheckBox, QPushButton,
    QTabWidget, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, Slot, QRect, QPoint, QSize
from PySide6.QtGui import QImage, QPainter, QPen, QColor

class VideoWidget(QWidget):
    roi_selected = Signal(list) 

    def __init__(self):
        super().__init__()
        self.image = None
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setStyleSheet("background-color: #000; border: 1px solid #444;")
        
        self.drawing_active = False
        self.is_dragging = False
        self.origin = QPoint()
        self.current_rect = QRect()
        self.pen = QPen(QColor(255, 0, 0))
        self.pen.setWidth(4)

    def set_drawing_mode(self, active):
        self.drawing_active = active
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)

    def set_frame(self, qt_img):
        self.image = qt_img
        self.update() 

    def paintEvent(self, event):
        painter = QPainter(self)
        if self.image:
            w, h = self.width(), self.height()
            scaled = self.image.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.off_x = (w - scaled.width()) // 2
            self.off_y = (h - scaled.height()) // 2
            self.scale_w = scaled.width()
            self.scale_h = scaled.height()
            painter.drawImage(self.off_x, self.off_y, scaled)
        else:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "Awaiting Kinect Stream...")

        if self.is_dragging and not self.current_rect.isNull():
            painter.setPen(self.pen)
            painter.drawRect(self.current_rect)

    def mousePressEvent(self, event):
        if self.drawing_active and event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.current_rect = QRect(self.origin, QSize())
            self.is_dragging = True
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing_active and self.is_dragging:
            self.current_rect = QRect(self.origin, event.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing_active and self.is_dragging:
            self.is_dragging = False
            self.drawing_active = False 
            self.setCursor(Qt.ArrowCursor)
            
            if self.image:
                rx = self.current_rect.x() - getattr(self, 'off_x', 0)
                ry = self.current_rect.y() - getattr(self, 'off_y', 0)
                rw = self.current_rect.width()
                rh = self.current_rect.height()
                sw = getattr(self, 'scale_w', 1)
                sh = getattr(self, 'scale_h', 1)
                
                factor_x = 640.0 / (sw if sw > 0 else 1)
                factor_y = 480.0 / (sh if sh > 0 else 1)
                
                final_roi = [
                    int(max(0, rx * factor_x)),
                    int(max(0, ry * factor_y)),
                    int(rw * factor_x),
                    int(rh * factor_y)
                ]
                print(f"ROI Captured: {final_roi}")
                self.roi_selected.emit(final_roi)
            self.current_rect = QRect() 
            self.update()

class GeoBoxDashboard(QMainWindow):
    class Signals(QWidget):
        sea_changed = Signal(float)
        flip_changed = Signal(bool)
        mode_preset = Signal(str)
        mode_custom = Signal(list)
        contours_toggled = Signal(bool)
        contour_interval = Signal(float)
        depth_range_update = Signal(int, int)
        
        # --- NEW: Auto Contrast Signal ---
        trigger_auto_level = Signal() 

        rain_changed = Signal(bool, float)
        evap_changed = Signal(float)
        clear_water = Signal()
        start_roi_select = Signal()
        calib_start = Signal()
        calib_next = Signal()
        calib_finish = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GeoBox Studio - Integrated Controller")
        self.resize(1100, 768)
        self.setStyleSheet("QMainWindow { background-color: #333; color: #ccc; }")
        self.signals = self.Signals()
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        sidebar = QWidget()
        sidebar.setFixedWidth(320)
        side_layout = QVBoxLayout(sidebar)
        
        gb_global = QGroupBox("Global Geometry")
        l_global = QVBoxLayout()
        self.sea_slider = QSlider(Qt.Horizontal)
        self.sea_slider.setRange(-50, 50)
        self.sea_slider.valueChanged.connect(lambda v: self.signals.sea_changed.emit(v/100.0))
        l_global.addWidget(QLabel("Sea Level Offset"))
        l_global.addWidget(self.sea_slider)
        self.chk_flip = QCheckBox("Invert Orientation (Fix Flip)")
        self.chk_flip.toggled.connect(self.signals.flip_changed.emit)
        l_global.addWidget(self.chk_flip)
        gb_global.setLayout(l_global)
        
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("QTabWidget::pane { border: 1px solid #555; }")

        t_vis = QWidget()
        l_vis = QVBoxLayout()
        l_vis.addWidget(QLabel("Color Theme:"))
        self.combo = QComboBox()
        self.combo.addItems(['terrain', 'ocean', 'gist_earth', 'viridis', 'magma'])
        self.combo.currentTextChanged.connect(self.signals.mode_preset.emit)
        l_vis.addWidget(self.combo)
        
        gb_depth = QGroupBox("Sand Height Sensitivity")
        ld = QVBoxLayout()
        self.lbl_min = QLabel("Lowest Point (Water): 800mm")
        self.sld_min = QSlider(Qt.Horizontal)
        self.sld_min.setRange(400, 1500)
        self.sld_min.setValue(800)
        self.lbl_max = QLabel("Highest Point (Mountain): 950mm")
        self.sld_max = QSlider(Qt.Horizontal)
        self.sld_max.setRange(400, 1500)
        self.sld_max.setValue(950)
        
        self.sld_min.valueChanged.connect(self._update_depth)
        self.sld_max.valueChanged.connect(self._update_depth)
        
        ld.addWidget(self.lbl_min)
        ld.addWidget(self.sld_min)
        ld.addWidget(self.lbl_max)
        ld.addWidget(self.sld_max)
        
        # --- NEW: Auto Level Button ---
        btn_auto = QPushButton("Auto-Level Contrast (Scan Sand)")
        btn_auto.setStyleSheet("background-color: #0066cc; color: white; padding: 5px;")
        btn_auto.clicked.connect(self.signals.trigger_auto_level.emit)
        ld.addWidget(btn_auto)

        gb_depth.setLayout(ld)
        l_vis.addWidget(gb_depth)
        
        gb_cont = QGroupBox("Topography")
        lc = QVBoxLayout()
        self.btn_cont = QPushButton("Toggle Contours")
        self.btn_cont.setCheckable(True)
        self.btn_cont.toggled.connect(self.signals.contours_toggled.emit)
        lc.addWidget(self.btn_cont)
        self.slider_cont = QSlider(Qt.Horizontal)
        self.slider_cont.setRange(2, 20)
        self.slider_cont.setValue(5)
        self.slider_cont.valueChanged.connect(lambda v: self.signals.contour_interval.emit(v/100.0))
        lc.addWidget(self.slider_cont)
        gb_cont.setLayout(lc)
        l_vis.addWidget(gb_cont)
        l_vis.addStretch()
        t_vis.setLayout(l_vis)
        
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

        t_calib = QWidget()
        l_calib = QVBoxLayout()
        l_calib.addWidget(QLabel("<b>Step 1: ROI</b>"))
        btn_roi = QPushButton("Draw Sand Box")
        btn_roi.clicked.connect(self.activate_drawing)
        l_calib.addWidget(btn_roi)
        l_calib.addWidget(QLabel("<b>Step 2: 10-Step Wizard</b>"))
        l_calib.addWidget(QLabel("Phase 1: Flat Sand (5 Steps)\nPhase 2: Board on Top (5 Steps)"))
        self.btn_start = QPushButton("Start Wizard")
        self.btn_start.setStyleSheet("background-color: #005; font-weight: bold;")
        self.btn_start.clicked.connect(self.signals.calib_start.emit)
        l_calib.addWidget(self.btn_start)
        self.btn_next = QPushButton("Capture & Next")
        self.btn_next.setStyleSheet("background-color: #050; font-weight: bold; height: 30px;")
        self.btn_next.clicked.connect(self.signals.calib_next.emit)
        l_calib.addWidget(self.btn_next)
        l_calib.addWidget(QLabel("<b>Step 3: Finish</b>"))
        btn_save = QPushButton("Compute & Save")
        btn_save.clicked.connect(self.signals.calib_finish.emit)
        l_calib.addWidget(btn_save)
        l_calib.addStretch()
        t_calib.setLayout(l_calib)

        self.tabs.addTab(t_vis, "Visuals")
        self.tabs.addTab(t_fluid, "Fluids")
        self.tabs.addTab(t_calib, "Calibration")

        side_layout.addWidget(gb_global)
        side_layout.addWidget(self.tabs)
        side_layout.addStretch()
        self.video = VideoWidget()
        layout.addWidget(sidebar)
        layout.addWidget(self.video, stretch=1)

    def activate_drawing(self):
        self.signals.start_roi_select.emit()
        self.video.set_drawing_mode(True)

    def _update_rain(self):
        enabled = self.chk_rain.isChecked()
        rate = self.sld_rain.value() / 1000.0
        self.signals.rain_changed.emit(enabled, rate)

    def _update_depth(self):
        mn = self.sld_min.value()
        mx = self.sld_max.value()
        if mn >= mx - 50:
             mn = mx - 50
             self.sld_min.setValue(mn)
        self.lbl_min.setText(f"Lowest Point (Water): {mn}mm")
        self.lbl_max.setText(f"Highest Point (Mountain): {mx}mm")
        self.signals.depth_range_update.emit(mn, mx)

    # --- NEW: Slot to receive Auto-Level updates from Engine ---
    @Slot(int, int)
    def set_depth_sliders(self, mn, mx):
        # Block signals to prevent feedback loop? No, valueChanged emits.
        # We just set them.
        self.sld_min.blockSignals(True)
        self.sld_max.blockSignals(True)
        self.sld_min.setValue(mn)
        self.sld_max.setValue(mx)
        self.sld_min.blockSignals(False)
        self.sld_max.blockSignals(False)
        self._update_depth() # Update labels and emit once

    @Slot(np.ndarray)
    def update_feed(self, frame):
        if len(frame.shape) == 2:
            h, w = frame.shape
            fmt = QImage.Format_Grayscale8
            bpl = w
        else:
            h, w, ch = frame.shape
            fmt = QImage.Format_BGR888
            bpl = ch * w
        img = QImage(frame.data, w, h, bpl, fmt)
        self.video.set_frame(img.copy())
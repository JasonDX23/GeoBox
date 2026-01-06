import numpy as np
import cv2
import sys
import os
from PySide6.QtWidgets import (QMainWindow, QLabel, QVBoxLayout, QWidget, QProgressBar,
                             QPushButton, QSlider, QHBoxLayout, QFrame, QComboBox, QApplication)
from PySide6.QtGui import QImage, QPixmap, QFont, QPen, QPainter
from PySide6.QtCore import Qt, Slot, Signal, QRect

from core.kinect import KinectWorker
from core.processor import TerrainProcessor, TerrainProcessor_Smoothened
from core.KinectProjector import KinectProjector

class GeoBox(QMainWindow):
    def __init__(self):
        super.__init__()
        self.setWindowTitle('GeoBox')
        self.resize(1200, 800)
        
        self.worker = KinectWorker()
        self.processor = TerrainProcessor()
        self.calibration = KinectProjector()
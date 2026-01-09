from PySide6.QtWidgets import QWidget, QRubberBand
from PySide6.QtCore import QRect, QSize, QPoint, Signal
from PySide6.QtGui import QPainter, QPen, QColor

class ROISelector(QWidget):
    roi_selected = Signal(list) # Emits [x, y, w, h]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.origin = QPoint()
        self.is_selecting = False
        # Make background transparent but capture mouse
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
        # Scale coords if the widget size != camera size (assuming 640x480 logic)
        # For simplicity, we assume the preview widget maintains aspect ratio
        self.roi_selected.emit([rect.x(), rect.y(), rect.width(), rect.height()])
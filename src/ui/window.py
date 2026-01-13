import cv2
import numpy as np
from PySide6.QtGui import QGuiApplication

class RenderWindow:
    def __init__(self, name="GeoBox Projector"):
        self.name = name
        self.screen_rect = None
        self.native_w = 1024    
        self.native_h = 768
        
        app = QGuiApplication.instance()
        screens = app.screens()
        
        # 1. Target the Projector
        if len(screens) > 1:
            target = screens[1]
            geom = target.geometry()
            self.screen_rect = (geom.x(), geom.y())
            self.native_w = geom.width()
            self.native_h = geom.height()
        else:
            geom = screens[0].geometry()
            self.screen_rect = (geom.x(), geom.y())
            # For primary screen debugging, keep standard size
            self.native_w = 1024
            self.native_h = 768

        # 2. Setup Window
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        
        # 3. Force Resize to Projector Native Resolution
        cv2.resizeWindow(self.name, self.native_w, self.native_h)
        
        # 4. Move and Fullscreen
        cv2.moveWindow(self.name, self.screen_rect[0], self.screen_rect[1])
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, frame):
        # OpenCV automatically scales the frame (640x480) 
        # to fill the Window Size (Projector Native)
        cv2.imshow(self.name, frame)

    def process_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): return False
        return True

    def destroy(self):
        cv2.destroyAllWindows()
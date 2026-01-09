import cv2
import numpy as np

class RenderWindow:
    def __init__(self, name="GeoBox AR"):
        self.name = name
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def show(self, frame):
        cv2.imshow(self.name, frame)

    def process_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'): # ESC or q
            return False
        return True

    def destroy(self):
        cv2.destroyAllWindows()
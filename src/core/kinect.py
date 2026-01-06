import os
current_folder = os.getcwd()
os.add_dll_directory(current_folder)

import freenect
import numpy as np
from PySide6.QtCore import QThread, Signal

class KinectWorker(QThread):
    depth_frame_ready = Signal(np.ndarray)
    #rgb_frame = Signal(np.ndarray)

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.running = True
        self.accumulator = None
        # # Define the physical workspace in mm
        # self.min_depth = 762   # Closest distance to sand (mm)
        # self.max_depth = 914  # Furthest distance (bottom of box) (mm)
        self.latest_rgb = None
        
        # Todo: DO WE NEED TO ADD THE ROI LOGIC HERE or in PROCESSOR.py?

    def run(self):
        while self.running:
            try:
                # Get registered depth (metric mm aligned to RGB/Projector space)
                depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
                rgb, _ = freenect.sync_get_video()
                if depth is None: continue
                
                # np.clip(depth, 0, 1023, out=depth) # Clipping not recommended
                # depth >>= 2
                current_frame = depth.astype(np.float32)

                # 3. Temporal Smoothing (EMA)
                if self.accumulator is None:
                    self.accumulator = current_frame
                else:
                    self.accumulator = (self.alpha * current_frame) + ((1.0 - self.alpha) * self.accumulator)

                self.depth_frame_ready.emit(self.accumulator.astype(np.uint8))
                self.latest_rgb = rgb

            except Exception as e:
                print(f"Kinect Sync Error: {e}")
                self.msleep(500)

    def get_latest_rgb(self):
        return self.latest_rgb if self.latest_rgb is not None else np.zeros((480, 640, 3), np.uint8)
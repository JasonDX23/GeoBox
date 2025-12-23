## ORIGINAL BASE CODE FOR KINECT PROCESSING - COMMENTED OUT BECAUSE IT DID NOT USE REGISTERED_DEPTH
# import freenect
# import numpy as np
# from PySide6.QtCore import QThread, Signal

# class KinectWorker(QThread):
#     depth_frame_ready = Signal(np.ndarray)

#     def __init__(self, alpha=0.3):
#         super().__init__()
#         self.alpha = alpha
#         self.running = True
#         self.accumulator = None

#     def run(self):
#         print("Kinect Thread Started: Initializing sensor...")
#         while self.running:
#             # 1. Fetch depth with a small timeout/retry logic
#             try:
#                 # Using the exact same acquisition logic as your working script
#                 depth, _ = freenect.sync_get_depth()
#                 if depth is None:
#                     continue
                
#                 # 2. Normalize to 8-bit (as done in your working notebook)
#                 # This ensures the processor.py receives data in the expected range
#                 np.clip(depth, 0, 1023, out=depth)
#                 depth >>= 2
#                 current_frame = depth.astype(np.float32)

#                 # 3. Smoothing (EMA)
#                 if self.accumulator is None:
#                     self.accumulator = current_frame
#                 else:
#                     self.accumulator = (self.alpha * current_frame) + \
#                                        ((1.0 - self.alpha) * self.accumulator)

#                 # 4. Emit the signal
#                 self.depth_frame_ready.emit(self.accumulator.astype(np.uint8))
            
#             except Exception as e:
#                 print(f"Kinect Sync Error: {e}")
#                 self.msleep(500) # Wait and retry

#     def stop(self):
#         self.running = False
#         self.wait()

import freenect
import numpy as np
from PySide6.QtCore import QThread, Signal

class KinectWorker(QThread):
    depth_frame_ready = Signal(np.ndarray)

    def __init__(self, alpha=0.3):
        super().__init__()
        self.alpha = alpha
        self.running = True
        self.accumulator = None
        # Define the physical workspace in mm
        self.min_depth = 762   # Closest distance to sand (mm)
        self.max_depth = 1016  # Furthest distance (bottom of box) (mm)

    def run(self):
        while self.running:
            try:
                # Get registered depth (metric mm aligned to RGB/Projector space)
                depth, _ = freenect.sync_get_depth(format=freenect.DEPTH_REGISTERED)
                if depth is None: continue

                # 1. Mask invalid pixels (freenect uses 2047 or 0 for no-data)
                depth = depth.astype(np.float32)
                depth[depth == 0] = self.max_depth
                depth[depth > 2047] = self.max_depth

                # 2. Normalize mm to 8-bit (0-255)
                # Instead of bit-shifting, we map our sandbox range to 0-255
                depth_norm = np.clip(depth, self.min_depth, self.max_depth)
                depth_norm = ((depth_norm - self.min_depth) / (self.max_depth - self.min_depth) * 255)
                
                # 3. Temporal Smoothing (EMA)
                if self.accumulator is None:
                    self.accumulator = depth_norm
                else:
                    self.accumulator = (self.alpha * depth_norm) + ((1.0 - self.alpha) * self.accumulator)

                self.depth_frame_ready.emit(self.accumulator.astype(np.uint8))
            
            except Exception as e:
                print(f"Kinect Sync Error: {e}")
                self.msleep(500)
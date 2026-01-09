import freenect
import numpy as np
import sys

class KinectDevice:
    def __init__(self):
        self.ctx = freenect.init()
        if freenect.num_devices(self.ctx) < 1:
            sys.stderr.write("Critical Error: No Kinect v1 device detected.\n")
            # In production, we might mock this, but for now we exit
            self.dummy = True
        else:
            self.dummy = False
            self.device = freenect.open_device(self.ctx, 0)
            freenect.set_depth_mode(self.device, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_11BIT)
            freenect.set_video_mode(self.device, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB)

    def get_depth_frame(self):
        if self.dummy: return np.zeros((480, 640), dtype=np.float32)
        depth, _ = freenect.sync_get_depth()
        if depth is None: raise RuntimeError("Depth stream fail")
        return depth.astype(np.float32)

    def get_rgb_frame(self):
        """Fetches standard RGB video frame for calibration detection"""
        if self.dummy: return np.zeros((480, 640, 3), dtype=np.uint8)
        rgb, _ = freenect.sync_get_video()
        if rgb is None: raise RuntimeError("Video stream fail")
        return rgb.astype(np.uint8) # RGB

    def close(self):
        if not self.dummy:
            freenect.close_device(self.device)
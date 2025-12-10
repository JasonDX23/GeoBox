import numpy as np
import freenect

class KinectV1:

    def __init__(self):
        self.name = 'kinect_v1'
        self.depth_width = 320
        self.depth_height = 240
        self.color_width = 640
        self.color_height = 480

        self.id = 0
        self.device = None
        self.depth = None
        self.color = None

        ctx = freenect.init()
        self.device = freenect.open_device(ctx, self.id)
        print(self.id)
        freenect.close_device(self.device)
        self.depth = self.get_frame()

    def get_frame(self):
        self.depth = freenect.sync_get_depth(index=self.id, format=freenect.DEPTH_MM)[0]
        self.depth = np.fliplr(self.depth) # Is this needed?
        return self.depth
    
    def get_color(self):
        self.color = freenect.sync_get_video(index=self.id)[0]
        self.color = np.fliplr(self.color) # Is this needed?
        return self.color
    
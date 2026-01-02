import cv2
import numpy as np

class KinectProjector:
    def __init__(self, projector_width=1024, projector_height=768):
        self.width = projector_width
        self.height = projector_height
        self._is_calibrated = False
        self._is_calibrating = False
        self._spatial_filtering = True 
        
        self.homography = None
        self.pattern_size = (5,4) # (Internal corners: width, height)
        
        self.collected_kinect_points = []
        self.collected_projector_points = []

    def setSpatialFiltering(self, enabled: bool):
        self._spatial_filtering = enabled

    def find_chessboard(self, frame):
        """Processes RGB frame to find high-contrast chessboard corners."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._spatial_filtering:
            # Gaussian is faster than Median for real-time detection
            gray = cv2.GaussianBlur(gray, (3, 3), 0)
            
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + 
                                                cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return None

    def generate_projector_points(self, off_x, off_y, sq_sz=80):
        points = []
        # Start at 1 to skip the outer edge and get the first internal intersection
        for y in range(1, 5): # 6 internal corners high
            for x in range(1, 6): # 7 internal corners wide
                points.append([off_x + x * sq_sz, off_y + y * sq_sz])
        return np.array(points, dtype=np.float32)

    # Inside KinectProjector_calibration.py
    def calculate_mapping(self):
        src = np.array(self.collected_kinect_points, dtype=np.float32)
        dst = np.array(self.collected_projector_points, dtype=np.float32)
        
        # RANSAC is better here because it will ignore "shaky" corner 
        # detections at the very edge of the projector beam.
        self.homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        if self.homography is not None:
            self._is_calibrated = True
            return True
        return False
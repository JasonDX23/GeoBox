# NEEDS TO BE IMPLEMENTED - 28/12/25

import cv2
import numpy as np

class KinectProjector:
    def __init__(self, projector_width=1024, projector_height=768):
        self.width = projector_width
        self.height = projector_height
        self._is_calibrated = False
        self._is_calibrating = False
        self._spatial_filtering = True # Default to ON for stability
        
        self.homography = None
        self.pattern_size = (7, 6)
        
        # Storage for the 60 required calibration points
        self.collected_kinect_points = []
        self.collected_projector_points = []

    def setSpatialFiltering(self, enabled: bool):
        self._spatial_filtering = enabled

    def find_chessboard(self, frame):
        # Apply spatial filtering if enabled to stabilize corner detection
        if self._spatial_filtering:
            frame = cv2.medianBlur(frame, 5)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, 
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        return None

    def calculate_mapping(self):
        """Finalizes calibration using internally collected points."""
        if len(self.collected_kinect_points) < 4:
            return False

        # findHomography maps Kinect (Source) -> Projector (Destination)
        self.homography, _ = cv2.findHomography(
            np.array(self.collected_kinect_points), 
            np.array(self.collected_projector_points)
        )
        self._is_calibrated = True
        self._is_calibrating = False
        return True
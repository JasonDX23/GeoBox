import cv2
import numpy as np

class DamController:
    def __init__(self):
        # 4x4_50 is standard and robust
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.params = cv2.aruco.DetectorParameters()
        
        # --- DAM SETTINGS ---
        # Adjust these to match your physical object size
        self.dam_length = 200  # Length of the wall in pixels
        self.dam_thickness = 25 # Thickness of the wall
        self.dam_height = 1.0  # 1.0 = Maximum height (blocks all water)

    def get_dam_mask(self, rgb_image, output_shape=(480, 640)):
        """
        Scans the RGB image for markers and returns a 'height map' 
        with walls drawn at the marker locations.
        """
        # Initialize empty mask (0 height)
        mask = np.zeros(output_shape, dtype=np.float32)
        
        if rgb_image is None:
            return mask

        # 1. Detect Markers
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        
        if ids is not None:
            for i, c in enumerate(corners):
                # ArUco corners: [TopLeft, TopRight, BotRight, BotLeft]
                pts = c[0]
                
                # 2. Calculate Center
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
                
                # 3. Calculate Rotation Angle
                # Vector from Top-Left to Top-Right
                dx = pts[1][0] - pts[0][0]
                dy = pts[1][1] - pts[0][1]
                angle_deg = np.degrees(np.arctan2(dy, dx))
                
                # 4. Draw the Dam (Rotated Rectangle)
                # We draw a high value (self.dam_height) onto the mask
                rect = ((cx, cy), (self.dam_length, self.dam_thickness), angle_deg)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Fill the rectangle with "Height"
                cv2.fillPoly(mask, [box], self.dam_height)
                
        return mask
import cv2
import numpy as np

class ProjectorCalibrator:
    def __init__(self):
        # The Coefficient Matrix (2x4) mapping [u, v, d, 1] -> [x, y]
        # Default: Identity-ish (x=u, y=v, d=0)
        self.matrix = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ], dtype=np.float32)
        
        self.roi = [0, 0, 640, 480]
        
        # Precomputed grid for fast remapping
        self.grid_u, self.grid_v = np.meshgrid(np.arange(640), np.arange(480))
        self.grid_u = self.grid_u.astype(np.float32)
        self.grid_v = self.grid_v.astype(np.float32)
        self.static_map_x = None
        self.static_map_y = None
        
        # Calibration State
        self.board_dims = (9, 6)
        self.collected_points_in = []  # [u, v, d]
        self.collected_points_out = [] # [x, y]

        # Generate Target Pattern Coordinates (Projector Space)
        self.target_pts = []
        sq_size = 50
        off_x, off_y = 50, 50
        for r in range(self.board_dims[1]):
            for c in range(self.board_dims[0]):
                px = off_x + (c+1)*sq_size
                py = off_y + (r+1)*sq_size
                self.target_pts.append([px, py])
        self.target_pts = np.array(self.target_pts, dtype=np.float32)

    def set_config(self, roi, matrix_list):
        self.roi = roi
        self.matrix = np.array(matrix_list, dtype=np.float32)
        self._precompute_static_maps()

    def _precompute_static_maps(self):
        """
        Precomputes the parts of the mapping that don't depend on depth.
        Map = C0*u + C1*v + C2*d + C3
        Static = C0*u + C1*v + C3
        """
        m = self.matrix
        # Vectorized calculation of static components
        self.static_map_x = (m[0,0] * self.grid_u + 
                             m[0,1] * self.grid_v + 
                             m[0,3])
                             
        self.static_map_y = (m[1,0] * self.grid_u + 
                             m[1,1] * self.grid_v + 
                             m[1,3])

    def reset_collection(self):
        self.collected_points_in = []
        self.collected_points_out = []

    def capture_points(self, rgb_frame, depth_frame):
        """
        Detects checkerboard and correlates RGB corners with Depth values.
        Returns: (Success, DebugImage)
        """
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        
        # Crop to ROI for reliability
        x, y, w, h = self.roi
        gray_roi = gray[y:y+h, x:x+w]
        
        found, corners = cv2.findChessboardCorners(gray_roi, self.board_dims, None)
        
        viz = rgb_frame.copy()
        
        if found:
            # Subpixel refinement
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray_roi, corners, (11,11), (-1,-1), term)
            
            # Draw on debug image (shift back to global coords)
            corners_viz = corners2.copy()
            corners_viz[:, 0, 0] += x
            corners_viz[:, 0, 1] += y
            cv2.drawChessboardCorners(viz, self.board_dims, corners_viz, found)
            
            # Extract data
            for i, pt in enumerate(corners2):
                u_local, v_local = pt[0]
                u_global, v_global = int(u_local + x), int(v_local + y)
                
                # Sample depth at this location
                # Safety check
                if 0 <= u_global < 640 and 0 <= v_global < 480:
                    d_val = depth_frame[v_global, u_global]
                    
                    # Ignore holes (0) or shadow errors (2047)
                    if 0 < d_val < 2047:
                        self.collected_points_in.append([u_global, v_global, d_val])
                        self.collected_points_out.append(self.target_pts[i])
            
            return True, viz
            
        return False, viz

    def compute_calibration(self):
        """
        Solves the Linear Regression for the 2.5D matrix.
        We want M such that: [x, y] = M * [u, v, d, 1]
        """
        if len(self.collected_points_in) < 20:
            print("Not enough points collected.")
            return False

        # Prepare matrices
        # A: Input [N, 4] -> u, v, d, 1
        A = np.hstack([np.array(self.collected_points_in), np.ones((len(self.collected_points_in), 1))])
        
        # B: Output [N, 2] -> x, y
        B = np.array(self.collected_points_out)
        
        # Solve Ax = B using Least Squares
        # Result X will be [4, 2]
        res, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        
        # Transpose to get [2, 4] matrix
        self.matrix = res.T.astype(np.float32)
        print("Calibration Matrix:\n", self.matrix)
        
        self._precompute_static_maps()
        return True

    def warp(self, depth_frame):
        """
        Fast 2.5D warping using precomputed maps + depth component.
        """
        if self.static_map_x is None: return depth_frame

        # Dynamic Component: C2 * d
        # Note: depth_frame must be float for calculation
        d_f = depth_frame.astype(np.float32)
        
        # Calculate full maps
        map_x = self.static_map_x + self.matrix[0, 2] * d_f
        map_y = self.static_map_y + self.matrix[1, 2] * d_f
        
        # Remap
        # INTER_NEAREST is faster and preserves depth integer values better than Linear
        warped = cv2.remap(depth_frame, map_x, map_y, cv2.INTER_NEAREST)
        
        return warped

    def generate_pattern(self):
        """Standard Checkerboard"""
        img = np.ones((480, 640, 3), dtype=np.uint8) * 255
        sq = 50
        ox, oy = 50, 50
        for r in range(self.board_dims[1] + 1):
            for c in range(self.board_dims[0] + 1):
                if (r+c)%2 == 1:
                    cv2.rectangle(img, (ox+c*sq, oy+r*sq), (ox+(c+1)*sq, oy+(r+1)*sq), (0,0,0), -1)
        return img
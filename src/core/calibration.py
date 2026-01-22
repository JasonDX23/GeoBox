import cv2
import numpy as np

class ProjectorCalibrator:
    def __init__(self, width=640, height=480):
        # Store projector dimensions to prevent broadcast errors
        self.w = width
        self.h = height
        
        # Matrix shape (2, 6) supports: [u, v, d, u*d, v*d, 1]
        self.matrix = np.zeros((2, 6), dtype=np.float32)
        self.matrix[0, 0] = 1.0 
        self.matrix[1, 1] = 1.0 
        
        self.roi = [0, 0, 640, 480]
        
        # Precompute Mapping Grids based on actual resolution
        self.grid_u, self.grid_v = np.meshgrid(np.arange(self.w), np.arange(self.h))
        self.grid_u = self.grid_u.astype(np.float32)
        self.grid_v = self.grid_v.astype(np.float32)
        
        # Checkerboard settings
        self.board_dims = (4, 3) 
        self.sq_size = 60 
        self.board_w = (self.board_dims[0] + 1) * self.sq_size
        self.board_h = (self.board_dims[1] + 1) * self.sq_size
        
        # Pattern Positions
        margin = 100
        min_cx, max_cx = (self.board_w // 2) + margin, self.w - (self.board_w // 2) - margin
        min_cy, max_cy = (self.board_h // 2) + margin, self.h - (self.board_h // 2) - margin
        
        self.positions = [
            (self.w // 2, self.h // 2), (min_cx, min_cy), (max_cx, min_cy), 
            (min_cx, max_cy), (max_cx, max_cy)
        ]
        
        self.collected_in = []
        self.collected_out = []
        self.current_step_pts = []

    def reset_collection(self):
        """Clears buffers for a fresh calibration run."""
        self.collected_in = []
        self.collected_out = []

    def set_config(self, roi, matrix_list):
        """
        Loads calibration data. 
        Handles migration from old 2x6 matrix to new 4x4 DLT matrix.
        """
        self.roi = roi
        
        # Load matrix
        try:
            mat = np.array(matrix_list, dtype=np.float32)
            
            # If we load a valid 4x4 matrix, use it
            if mat.shape == (4, 4):
                self.proj_matrix = mat
            else:
                # If shape is wrong (e.g. old (2,6) config), reset to Default
                print("Warning: Loaded config is incompatible (old format). Resetting to Identity.")
                self.proj_matrix = np.eye(4, dtype=np.float32)
                
        except Exception as e:
            print(f"Error loading config: {e}. Resetting to Identity.")
            self.proj_matrix = np.eye(4, dtype=np.float32)

    def generate_dynamic_pattern(self, step_idx):
        """Draws target on projector window."""
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        pos_idx = step_idx % 5
        cx, cy = self.positions[pos_idx]
        ox, oy = int(cx - self.board_w // 2), int(cy - self.board_h // 2)
        
        self.current_step_pts = []
        cv2.rectangle(img, (ox-15, oy-15), (ox+self.board_w+15, oy+self.board_h+15), (180, 180, 180), -1)
        
        for r in range(self.board_dims[1]): 
            for c in range(self.board_dims[0]):
                px, py = ox + (c+1)*self.sq_size, oy + (r+1)*self.sq_size
                self.current_step_pts.append([px, py])

        for r in range(self.board_dims[1]+1):
            for c in range(self.board_dims[0]+1):
                if (r+c)%2 == 1:
                    x1, y1 = ox + c*self.sq_size, oy + r*self.sq_size
                    cv2.rectangle(img, (x1, y1), (x1+self.sq_size, y1+self.sq_size), (0,0,0), -1)
        return img

    def get_depth_safe(self, depth_frame, u, v, step_idx):
        """Safe depth lookup."""
        if u < 2 or u >= 638 or v < 2 or v >= 478: return 0
        window = depth_frame[v-2:v+3, u-2:u+3]
        valid = window[(window > 0) & (window < 2047)]
        if len(valid) > 0: return np.median(valid)
        return 762.0 if step_idx >= 5 else 0

    def capture_frame(self, rgb, depth, step_idx):
        """Kinect vision processing."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        rx, ry, rw, rh = self.roi
        if rw < 10: return False, rgb
        
        gray_roi = cv2.equalizeHist(gray[ry:ry+rh, rx:rx+rw])
        found, corners = cv2.findChessboardCorners(gray_roi, self.board_dims, None)
        
        viz = rgb.copy()
        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray_roi, corners, (11,11), (-1,-1), term)
            batch_in, batch_out = [], []
            for i, pt in enumerate(corners2):
                u_g, v_g = int(pt[0][0] + rx), int(pt[0][1] + ry)
                d = self.get_depth_safe(depth, u_g, v_g, step_idx)
                if d > 0:
                    cv2.circle(viz, (u_g, v_g), 5, (0, 255, 0), -1)
                    batch_in.append([u_g, v_g, d])
                    batch_out.append(self.current_step_pts[i])
            if len(batch_in) > 3:
                self.collected_in.extend(batch_in)
                self.collected_out.extend(batch_out)
                return True, viz
        return False, viz

    def compute(self):
        """Solves 6-parameter projective parallax."""
        if len(self.collected_in) < 15: return False
        pts_in, pts_out = np.array(self.collected_in), np.array(self.collected_out)
        u, v, d = pts_in[:,0], pts_in[:,1], pts_in[:,2]

        # Design Matrix: [u, v, d, u*d, v*d, 1]
        A = np.column_stack([u, v, d, u*d, v*d, np.ones_like(u)])
        res, _, _, _ = np.linalg.lstsq(A, pts_out, rcond=None)
        
        self.matrix = res.T.astype(np.float32)
        return True

    def warp(self, depth):
        """Warps Kinect depth into Projector space."""
        d = depth.astype(np.float32)
        d[d >= 2047] = 0 # Mask shadows
        
        # Apply 6-parameter transform using indices 0-5
        mx = (self.matrix[0,0]*self.grid_u + self.matrix[0,1]*self.grid_v + 
              self.matrix[0,2]*d + self.matrix[0,3]*self.grid_u*d + 
              self.matrix[0,4]*self.grid_v*d + self.matrix[0,5])
              
        my = (self.matrix[1,0]*self.grid_u + self.matrix[1,1]*self.grid_v + 
              self.matrix[1,2]*d + self.matrix[1,3]*self.grid_u*d + 
              self.matrix[1,4]*self.grid_v*d + self.matrix[1,5])
        
        return cv2.remap(depth, mx, my, cv2.INTER_LINEAR, borderValue=0)
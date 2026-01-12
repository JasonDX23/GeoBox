import cv2
import numpy as np

class ProjectorCalibrator:
    def __init__(self):
        # We use a Linear Matrix (2x4) which is much more robust than Quadratic.
        # [C0*u + C1*v + C2*d + C3]
        self.matrix = np.zeros((2, 4), dtype=np.float32)
        # Default to Identity (1:1 mapping)
        self.matrix[0, 0] = 1.0 
        self.matrix[1, 1] = 1.0 
        
        self.roi = [0, 0, 640, 480]
        
        # Precomputed Mapping Grids
        self.grid_u, self.grid_v = np.meshgrid(np.arange(640), np.arange(480))
        self.grid_u = self.grid_u.astype(np.float32)
        self.grid_v = self.grid_v.astype(np.float32)
        
        # --- GEOMETRY SETTINGS ---
        # 4x3 Inner Corners = 5x4 Physical Squares
        # Large 100px squares for better visibility
        self.board_dims = (4, 3) 
        self.sq_size = 100 
        
        self.board_w = (self.board_dims[0] + 1) * self.sq_size
        self.board_h = (self.board_dims[1] + 1) * self.sq_size
        
        # Safe Positions (Calculated to keep pattern fully on screen)
        margin = 30
        min_cx = (self.board_w // 2) + margin
        max_cx = 640 - (self.board_w // 2) - margin
        min_cy = (self.board_h // 2) + margin
        max_cy = 480 - (self.board_h // 2) - margin
        
        self.positions = [
            (320, 240),       # Center
            (min_cx, min_cy), # Top-Left
            (max_cx, min_cy), # Top-Right
            (min_cx, max_cy), # Bot-Left
            (max_cx, max_cy)  # Bot-Right
        ]
        
        self.collected_in = []
        self.collected_out = []
        self.current_step_pts = []

    def set_config(self, roi, matrix_list):
        self.roi = roi
        mat = np.array(matrix_list, dtype=np.float32)
        
        # If loading an old Quadratic matrix (2x7), force reset to Linear (2x4)
        if mat.shape == (2, 7):
            print("[Calib] Converting old Quadratic config to Linear.")
            self.matrix = np.zeros((2, 4), dtype=np.float32)
            self.matrix[0, 0] = 1.0
            self.matrix[1, 1] = 1.0
        else:
            self.matrix = mat

    def reset_collection(self):
        self.collected_in = []
        self.collected_out = []

    def generate_dynamic_pattern(self, step_idx):
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        pos_idx = step_idx % 5
        if pos_idx >= len(self.positions): return img
            
        cx, cy = self.positions[pos_idx]
        ox = int(cx - self.board_w // 2)
        oy = int(cy - self.board_h // 2)
        
        self.current_step_pts = []
        
        # --- ANTI-GLARE BACKGROUND ---
        # Use Light Gray (180) instead of White (255) to prevent bloom
        bg_color = (180, 180, 180)
        
        pad = 20
        cv2.rectangle(img, (ox-pad, oy-pad), (ox+self.board_w+pad, oy+self.board_h+pad), bg_color, -1)
        
        # Generate Target Points
        for r in range(self.board_dims[1]): 
            for c in range(self.board_dims[0]):
                px = ox + (c+1)*self.sq_size
                py = oy + (r+1)*self.sq_size
                self.current_step_pts.append([px, py])

        # Draw Checkerboard Squares
        for r in range(self.board_dims[1]+1):
            for c in range(self.board_dims[0]+1):
                if (r+c)%2 == 1:
                    x1 = ox + c*self.sq_size
                    y1 = oy + r*self.sq_size
                    cv2.rectangle(img, (x1, y1), (x1+self.sq_size, y1+self.sq_size), (0,0,0), -1)
        
        # Draw "Coverage Map" (Tiny Green dots for collected points)
        for pt in self.collected_out:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 255, 0), -1)
            
        return img

    def get_depth_safe(self, depth_frame, u, v, step_idx):
        """
        Retrieves depth with multiple safety fallbacks.
        1. Checks neighbors (5x5) for salt-and-pepper noise.
        2. If in Phase 2 (Board) and sensor is blind, returns hardcoded 30 inches.
        """
        # 1. Bounds Check
        if u < 2 or u >= 638 or v < 2 or v >= 478: return 0
        
        # 2. Neighbor Search
        window = depth_frame[v-2:v+3, u-2:u+3]
        valid = window[(window > 0) & (window < 2047)]
        
        if len(valid) > 0:
            return np.median(valid)
        
        # 3. BLIND SPOT FIX (The Magic Sand Logic)
        # If we are in Phase 2 (Steps 6-10) and the sensor sees 0, 
        # assume it's the whiteboard at 30 inches (762mm).
        if step_idx >= 5:
            return 762.0 
            
        return 0

    def capture_frame(self, rgb, depth, step_idx):
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        x, y, w, h = self.roi
        
        if w < 10: return False, rgb
        
        # ROI Crop
        x, y = max(0, x), max(0, y)
        w, h = min(w, 640-x), min(h, 480-y)
        gray_roi = gray[y:y+h, x:x+w]
        
        # Contrast Boost
        gray_roi = cv2.equalizeHist(gray_roi)
        
        # Detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        found, corners = cv2.findChessboardCorners(gray_roi, self.board_dims, flags)
        
        viz = rgb.copy()
        
        if found:
            # Subpixel Refinement
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray_roi, corners, (11,11), (-1,-1), term)
            
            c_viz = corners2.copy()
            c_viz[:,0,0] += x
            c_viz[:,0,1] += y
            cv2.drawChessboardCorners(viz, self.board_dims, c_viz, found)
            
            batch_in = []
            batch_out = []
            
            for i, pt in enumerate(corners2):
                u, v = pt[0]
                u_g, v_g = int(u + x), int(v + y)
                
                # Use Safe Depth Lookup
                d = self.get_depth_safe(depth, u_g, v_g, step_idx)
                
                if d > 0:
                    cv2.circle(viz, (u_g, v_g), 5, (0, 255, 0), -1) # Green = Good
                    if i < len(self.current_step_pts):
                        batch_in.append([u_g, v_g, d])
                        batch_out.append(self.current_step_pts[i])
                else:
                    cv2.circle(viz, (u_g, v_g), 5, (0, 0, 255), -1) # Red = Bad

            # Accept if we have > 3 valid points (Relaxed constraint)
            if len(batch_in) > 3:
                self.collected_in.extend(batch_in)
                self.collected_out.extend(batch_out)
                return True, viz
                
        return False, viz

    def compute(self):
        if len(self.collected_in) < 15: 
            print("[Calib] Error: Not enough points collected (<15).")
            return False
        
        pts_in = np.array(self.collected_in)
        pts_out = np.array(self.collected_out)
        
        u, v, d = pts_in[:,0], pts_in[:,1], pts_in[:,2]
        
        # --- SPREAD CHECK ---
        # Prevents "Degenerate Matrix" by ensuring points cover the screen
        spread_u = np.max(u) - np.min(u)
        spread_v = np.max(v) - np.min(v)
        print(f"[Calib] Coverage Spread: W={spread_u:.0f}, H={spread_v:.0f}")
        
        if spread_u < 100 or spread_v < 100:
            print("[Calib] FAILURE: Points too clumped. Board reflection might be blinding camera.")
            return False

        # --- LINEAR SOLVE ---
        # x = C0*u + C1*v + C2*d + C3
        A = np.column_stack([u, v, d, np.ones_like(u)])
        
        res, _, _, _ = np.linalg.lstsq(A, pts_out, rcond=None)
        
        # Validate Coefficients
        c0, c1 = abs(res[0,0]), abs(res[1,1])
        print(f"[Calib] Scale Check: X={c0:.3f}, Y={c1:.3f}")
        
        if c0 < 0.1 or c1 < 0.1:
            print("[Calib] FAILURE: Matrix collapsed (Scale near 0).")
            return False
            
        self.matrix = res.T.astype(np.float32)
        print("[Calib] Linear Calibration SUCCESS.")
        return True

    def warp(self, depth):
        d = depth.astype(np.float32)
        
        # Apply 2x4 Linear Matrix
        mx = (self.matrix[0,0]*self.grid_u + self.matrix[0,1]*self.grid_v + 
              self.matrix[0,2]*d + self.matrix[0,3])
              
        my = (self.matrix[1,0]*self.grid_u + self.matrix[1,1]*self.grid_v + 
              self.matrix[1,2]*d + self.matrix[1,3])
        
        # Use Linear Interpolation for smooth result
        return cv2.remap(depth, mx, my, cv2.INTER_LINEAR, borderValue=0)
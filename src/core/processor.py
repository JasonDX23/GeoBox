## BETA 1.0 - I really liked this one but the depth shadow was an issue

import numpy as np
import cv2

class TerrainProcessor:
    def __init__(self):
        self.base_plane = None  # Will store the "empty sandbox" depth map
        # Standard Kinect v1 raw-to-metric approximation if needed:
        # depth_m = 1.0 / (raw_depth * -0.0030711016 + 3.3309495161)
        self.plane_params = None
        self.fx = 580.0 
        self.fy = 580.0
        self.cx = 320.0
        self.cy = 240.0
        y, x = np.indices((480, 640))
        self.x_multiplier = (x - self.cx) / self.fx
        self.y_multiplier = (y - self.cy) / self.fy

    def depth_to_world(self, depth_map):
        """Converts depth map to (N, 3) point cloud in millimeters."""
        h, w = depth_map.shape
        v, u = np.mgrid[0:h, 0:w]
        
        # Filter zero/invalid depth
        mask = depth_map > 0
        z = depth_map[mask].astype(float)
        x = (u[mask] - self.cx) * z / self.fx
        y = (v[mask] - self.cy) * z / self.fy
        
        return np.column_stack((x, y, z))
    
    def depth_to_world_cropped(self, cropped_depth):
        # Use the pre-sliced multipliers for consistent 3D points
        mask = cropped_depth > 0
        z = cropped_depth[mask].astype(float)
        x = self.x_multiplier[mask] * z
        y = self.y_multiplier[mask] * z
        return np.column_stack((x, y, z))

    def set_base_plane(self, flat_sand_frame):
        # Ensure we only fit the plane to sand inside the ROI
        if hasattr(self, 'roi') and self.roi is not None:
            x, y, w, h = self.roi
            flat_sand_frame = flat_sand_frame[y:y+h, x:x+w]
        """Fits a plane using memory-efficient Covariance + SVD."""
        points = self.depth_to_world(flat_sand_frame)
        centroid = np.mean(points, axis=0)
        pts_centered = points - centroid

        # Instead of SVD on (N, 3), use SVD on the (3, 3) covariance matrix
        # This avoids the N x N memory allocation entirely
        cov = np.dot(pts_centered.T, pts_centered) 
        _, _, vh = np.linalg.svd(cov)
        
        # The normal is the eigenvector corresponding to the smallest eigenvalue
        # In the SVD of the covariance matrix, this is the last row of Vh
        normal = vh[2, :] 
        
        # Ensure the normal points 'up' toward the sensor
        if normal[2] < 0:
            normal = -normal

        d = -np.dot(normal, centroid)
        self.plane_params = np.append(normal, d)

    def update_roi(self, x, y, w, h):
        self.roi = (x, y, w, h)
        # Re-slice the multipliers to match the new window
        # This keeps the 3D projection mathematically sound
        full_y, full_x = np.indices((480, 640))
        self.x_multiplier = ((full_x[y:y+h, x:x+w] - self.cx) / self.fx).astype(np.float32)
        self.y_multiplier = ((full_y[y:y+h, x:x+w] - self.cy) / self.fy).astype(np.float32)

    def calculate_elevation(self, raw_frame):
        if self.plane_params is None or self.roi is None:
            return np.zeros_like(raw_frame, dtype=np.float32)
            
        x_roi, y_roi, w, h = self.roi
        z = raw_frame[y_roi:y_roi+h, x_roi:x_roi+w].astype(np.float32)
        
        a, b, c, d = self.plane_params
        
        # Calculate perpendicular distance
        current_dist = (a * self.x_multiplier * z) + (b * self.y_multiplier * z) + (c * z) + d
        
        # CHANGE THIS LINE: 
        # If -current_dist was blue peaks, use current_dist
        elevation = -current_dist 
        
        # Clip to a realistic range (0 to 150mm)
        return np.clip(elevation, 0, 150)
    
    def apply_color_map(self, elevation):
        # Set this to the physical depth of your box in millimeters
        # Any sand higher than this will be pure white (255)
        # The floor will be 0 (Blue)
        max_height_mm = 254
        
        norm_depth = (elevation / max_height_mm) * 255
        return np.clip(norm_depth, 0, 255).astype(np.uint8)
    
    def process_frame(self, raw_frame):
        # 1. Calculate height
        elev = self.calculate_elevation(raw_frame)
        # 2. Spatial smoothing
        elev_smoothed = cv2.GaussianBlur(elev, (5, 5), 0)
        # 3. Colorize
        rgb_terrain = self.apply_color_map(elev_smoothed)
        return rgb_terrain
    
    def generate_contours(elevation_map, interval=50, thickness=1):
        """
        Creates a binary mask of topographic lines.
        elevation_map: 2D float32 array from calculate_elevation()
        interval: Distance in raw units between each line
        """
        # 1. Apply modulo to find the remainder relative to the interval
        # This creates a 'sawtooth' pattern across the terrain
        mod_map = np.mod(elevation_map, interval)
        
        # 2. Threshold the remainder to find the 'edges' of each interval
        # We look for values very close to the interval boundary
        contour_mask = np.where(mod_map < thickness, 255, 0).astype(np.uint8)
        
        # 3. Clean up the mask using a morphological opening to remove noise
        kernel = np.ones((3,3), np.uint8)
        contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, kernel)
        
        return contour_mask
    
    def get_clean_contours(self, elevation_map, interval=20):
        # 1. Smoothing is vital before contouring to stop "wobbly" lines
        smoothed = cv2.GaussianBlur(elevation_map, (7, 7), 0)
        
        # 2. Quantize
        quantized = (smoothed // interval) * interval
        
        # 3. Use Laplacian or Canny to find the steps
        edges = cv2.Canny(quantized.astype(np.uint8), 1, 1)
        
        # 4. Thicken the lines slightly for better projection visibility
        kernel = np.ones((2,2), np.uint8)
        return cv2.dilate(edges, kernel, iterations=1)
    
    def overlay_contours(rgb_image, contour_mask):
        # Make contours black (0,0,0) where the mask is active
        rgb_image[contour_mask > 0] = [0, 0, 0] 
        return rgb_image
    
    def get_slopes(self, elevation):
        # Sobel filters find the intensity gradient in X and Y directions
        dx = cv2.Sobel(elevation, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(elevation, cv2.CV_32F, 0, 1, ksize=3)
        return dx, dy


## BETA 3 with filtering algorithms

from collections import deque

class TerrainProcessor_Smoothened:
    def __init__(self, base_plane_path=None, window_size=20, hysteresis_threshold=2.0):
        self.base_plane = None 

        self.frame_buffer = deque(maxlen=window_size)
        self.stable_depth = None
        self.h_threshold = hysteresis_threshold
        # Standard Kinect v1 raw-to-metric approximation if needed:
        # depth_m = 1.0 / (raw_depth * -0.0030711016 + 3.3309495161)

    def set_base_plane(self, frame):
        smoothed_base = cv2.bilateralFilter(frame.astype(np.float32), 9, 75, 75)
        """Captures the current frame as the 'zero' elevation reference."""
        self.base_plane = smoothed_base

    def _apply_filters(self, raw_frame):
        self.frame_buffer.append(raw_frame.astype(np.float32))
        avg_frame = np.mean(self.frame_buffer, axis=0)

        # B. Hysteresis Envelope (Prevents pixel 'boiling')
        if self.stable_depth is None:
            self.stable_depth = avg_frame
        else:
            # Only update pixels that have moved more than the threshold
            diff = np.abs(avg_frame - self.stable_depth)
            update_mask = diff > self.h_threshold
            self.stable_depth[update_mask] = avg_frame[update_mask]

        # C. Spatial Smoothing & Shadow Filling
        # Bilateral filter preserves edges while smoothing the flat sand
        filtered = cv2.bilateralFilter(self.stable_depth, 5, 50, 50)
        
        # Fill Kinect Shadows (0 values) with background/average
        mask = (filtered <= 0).astype(np.uint8)
        if np.any(mask):
            filtered = cv2.inpaint(filtered, mask, 3, cv2.INPAINT_TELEA)
            
        return filtered

    def calculate_elevation(self, current_frame):
        if self.base_plane is None:
            return np.zeros_like(current_frame)
        
        # Apply filter pipeline before calculation
        filtered_depth = self._apply_filters(current_frame)
        
        # Elevation = Base Distance - Current Distance
        elevation = np.clip(self.base_plane - filtered_depth, 0, 1000)
        return elevation
    
    def apply_color_map(self, elevation):
        """Maps elevation values to RGB colors using a custom LUT."""
        # Normalize elevation to 0-255 for 8-bit color mapping
        max_height = np.max(elevation) if np.max(elevation) > 0 else 1.0
        norm_depth = (elevation / max_height * 255).astype(np.uint8)

        # Create a custom 256-entry gradient (Simplified Example)
        # In practice, you would define specific 'stops' for sea, land, mountain
        color_map = cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
        
        return color_map
    
    def process_frame(self, raw_frame):
        # 1. Calculate height
        elev = self.calculate_elevation(raw_frame)
        # 2. Spatial smoothing
        elev_smoothed = cv2.GaussianBlur(elev, (5, 5), 0)
        # 3. Colorize
        rgb_terrain = self.apply_color_map(elev_smoothed)
        return rgb_terrain
    
    def generate_contours(elevation_map, interval=50, thickness=2):
        """
        Creates a binary mask of topographic lines.
        elevation_map: 2D float32 array from calculate_elevation()
        interval: Distance in raw units between each line
        """
        # 1. Apply modulo to find the remainder relative to the interval
        # This creates a 'sawtooth' pattern across the terrain
        mod_map = np.mod(elevation_map, interval)
        
        # 2. Threshold the remainder to find the 'edges' of each interval
        # We look for values very close to the interval boundary
        contour_mask = np.where(mod_map < thickness, 255, 0).astype(np.uint8)
        
        # 3. Clean up the mask using a morphological opening to remove noise
        kernel = np.ones((3,3), np.uint8)
        contour_mask = cv2.morphologyEx(contour_mask, cv2.MORPH_OPEN, kernel)
        
        return contour_mask
    
    def get_clean_contours(elevation_map, interval=50):
        # Quantize the elevation (create 'stairs' in the terrain)
        quantized = (elevation_map // interval) * interval
        # Convert to 8-bit for OpenCV processing
        quantized_8 = np.clip(quantized, 0, 255).astype(np.uint8)
        # Detect edges between the 'stairs'
        edges = cv2.Canny(quantized_8, 1, 1)
        return edges
    
    def overlay_contours(rgb_image, contour_mask):
        # Make contours black (0,0,0) where the mask is active
        rgb_image[contour_mask > 0] = [0, 0, 0] 
        return rgb_image
    
    def get_slopes(self, elevation):
        # Sobel filters find the intensity gradient in X and Y directions
        dx = cv2.Sobel(elevation, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(elevation, cv2.CV_32F, 0, 1, ksize=3)
        return dx, dy


## TESTING CODE
# import cv2
# import numpy as np

# class TerrainProcessor:
#     def __init__(self, registration_table=None, depth_to_rgb_shift=None):
#         self.base_plane = None
#         # Tables for low-level registration (pre-calculate or load from calibration)
#         self.reg_table = registration_table  # u,v rectification table [cite: 21344]
#         self.shift_table = depth_to_rgb_shift # Depth-to-RGB shift [cite: 21342]
#         self.REG_SCALE = 256 

#     def register_frame(self, raw_frame):
#         """
#         Applies registration to align depth data and remove IR camera offsets.
#         Based on theoretical registration logic from libfreenect[cite: 21350].
#         """
#         h, w = raw_frame.shape
#         registered = np.zeros((h, w), dtype=np.float32)
        
#         # Iterate and apply shifts (optimized via vectorization in practice) [cite: 21352]
#         for v in range(h):
#             for u in range(w):
#                 raw_val = raw_frame[v, u]
#                 if raw_val >= 2047: continue # Skip 'no-depth' values [cite: 21340]
                
#                 # 1. Convert raw depth to metric Z (mm) [cite: 21386]
#                 metric_z = self._raw_to_mm(raw_val)
                
#                 # 2. Calculate rectified coordinates using shift tables [cite: 21357]
#                 # Horizontal shift is depth-dependent [cite: 21344]
#                 shift_x = self.shift_table[int(metric_z)]
#                 reg_u = int((self.reg_table[v, u, 0] + shift_x) / self.REG_SCALE)
#                 reg_v = int(self.reg_table[v, u, 1])
                
#                 # 3. Z-buffer check: store the closest depth value [cite: 21360]
#                 if 0 <= reg_u < w and 0 <= reg_v < h:
#                     if registered[reg_v, reg_u] == 0 or metric_z < registered[reg_v, reg_u]:
#                         registered[reg_v, reg_u] = metric_z
#         return registered

#     def _raw_to_mm(self, raw):
#         """Standard raw-to-metric conversion formula[cite: 21386]."""
#         # Example constants; in practice, use device-specific calibration [cite: 25148]
#         return 10.0 * (raw * -0.00307 + 3.33)

#     def fill_missing_data(self, depth_frame):
#         """Fills Kinect depth shadows (0 values) using fast interpolation."""
#         # Create a mask where depth is 0 (shadows/occlusions)
#         mask = (depth_frame == 0).astype(np.uint8)
        
#         # Optimization: Early exit if no shadows are detected
#         if not np.any(mask):
#             return depth_frame

#         # Inpaint works best on 8-bit or 16-bit. Since we normalized 
#         # to 0-255 in the worker, uint8 is perfect here.
#         depth_8 = depth_frame.astype(np.uint8)
#         fixed_8 = cv2.inpaint(depth_8, mask, 3, cv2.INPAINT_TELEA)
        
#         return fixed_8.astype(np.float32)

#     def set_base_plane(self, frame):
#         """Captures and fixes shadows for the 'zero' reference frame."""
#         # We sanitize the base plane so it's a solid surface without holes
#         fixed_frame = self.fill_missing_data(frame.astype(np.float32))
#         self.base_plane = fixed_frame

#     def calculate_elevation(self, current_frame):
#         """Calculates height after filling sensor shadows."""
#         # 1. Fill shadows in the incoming live frame
#         fixed_frame = self.fill_missing_data(current_frame.astype(np.float32))
        
#         if self.base_plane is None:
#             return np.zeros_like(fixed_frame)
        
#         # 2. Elevation = Base Distance - Current Distance
#         # Clipped at 255 to stay within 8-bit range for standard ARS setups
#         elevation = np.clip(self.base_plane - fixed_frame, 0, 255)
#         return elevation

#     def apply_color_map(self, elevation):
#         """Maps elevation values to RGB colors."""
#         # Normalize to ensure full use of the JET gradient
#         max_h = np.max(elevation) if np.max(elevation) > 0 else 1.0
#         norm_depth = (elevation / max_h * 255).astype(np.uint8)
#         return cv2.applyColorMap(norm_depth, cv2.COLORMAP_JET)
    
#     def process_frame(self, raw_frame):
#         # Apply low-level registration first to fix offsets
#         reg_depth = self.register_frame(raw_frame)
#         # Proceed with elevation calculation on registered data
#         elev = self.calculate_elevation(reg_depth)
#         elev_smoothed = cv2.GaussianBlur(elev, (5, 5), 0)
#         return self.apply_color_map(elev_smoothed)
    
#     def get_clean_contours(self, elevation_map, interval=50):
#         """Creates sharp topographic lines via quantization."""
#         if interval < 1: interval = 1
#         quantized = (elevation_map // interval) * interval
#         quantized_8 = np.clip(quantized, 0, 255).astype(np.uint8)
#         edges = cv2.Canny(quantized_8, 1, 1)
#         return edges
    
#     def get_slopes(self, elevation):
#         """Calculates gradients for rain physics."""
#         dx = cv2.Sobel(elevation, cv2.CV_32F, 1, 0, ksize=3)
#         dy = cv2.Sobel(elevation, cv2.CV_32F, 0, 1, ksize=3)
#         return dx, dy
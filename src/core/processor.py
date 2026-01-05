## BETA 1.0 - I really liked this one but the depth shadow was an issue

import numpy as np
import cv2
import matplotlib.pyplot as plt
class TerrainProcessor:
    def __init__(self):
        self.base_depth = None  # Stores the "Empty Box" snapshot
        self.roi = None         # (x, y, w, h)

        #for the terrain colormap
        colormap = plt.get_cmap('terrain')
        colors = colormap(np.linspace(0,1, 256))[:, :3]
        colors = (colors*255).astype(np.uint8)
        self.terrain_lut = colors[:, ::-1].reshape(256, 1, 3)
        
    def set_base_depth(self, frame):
        """Take a snapshot of the flat sand to use as 'Sea Level'"""
        self.base_depth = frame.astype(np.float32)

    def get_elevation(self, current_frame):
        """Calculates height by subtracting current sand from the floor."""
        if self.base_depth is None:
            return np.zeros_like(current_frame)

        # Convert to float for math
        curr = current_frame.astype(np.float32)
        
        # Height = Floor Depth - Current Depth
        # (Example: Floor is 900mm away, Sand is 800mm away -> Height is 100mm)
        elevation = self.base_depth - curr
        
        # Remove noise: anything less than 0 is just sensor error
        elevation[elevation < 0] = 0
        return elevation

    def process_frame(self, raw_frame):
        """The main pipeline called by your GUI"""
        # 1. Get height
        elev = self.get_elevation(raw_frame)
        
        # 2. Smooth the sand (Crucial for clean projection)
        # Replaces C++ 'applySpaceFilter'
        elev = cv2.GaussianBlur(elev, (7, 7), 0)

        # 3. Create Color Map (Hypsometric Tinting)
        # Replaces the complex C++ shader logic
        color_map = self.create_topo_map(elev)
        
        return color_map

    def create_topo_map(self, elevation):
        """Turns millimeters into a 0-255 image for the projector"""
        # Assume max sand height is 200mm
        max_h = 200.0
        norm = np.clip(elevation / max_h, 0, 1) * 255
        norm = norm.astype(np.uint8)
        
        # Apply a 'Terrain' color palette (Blue -> Green -> Brown -> White)
        # COLORMAP_TURBO or COLORMAP_terrain look great for sandboxes
        colored = cv2.LUT(cv2.merge([norm, norm, norm]), self.terrain_lut)

        # Alternative: If norm is a single channel, cv2.LUT needs a specific trick:
        # # colored = cv2.applyColorMap(norm, self.terrain_lut) 
        # # (OpenCV allows passing a custom LUT directly into applyColorMap too!)
        # colored = cv2.applyColorMap(norm, self.terrain_lut)
        
        return colored


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
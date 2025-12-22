## BETA 1.0 - I really liked this one but the depth shadow was an issue

import numpy as np
import cv2

class TerrainProcessor:
    def __init__(self, base_plane_path=None):
        self.base_plane = None  # Will store the "empty sandbox" depth map
        # Standard Kinect v1 raw-to-metric approximation if needed:
        # depth_m = 1.0 / (raw_depth * -0.0030711016 + 3.3309495161)

    def set_base_plane(self, frame):
        """Captures the current frame as the 'zero' elevation reference."""
        self.base_plane = frame.astype(np.float32)

    def calculate_elevation(self, current_frame):
        """Calculates height of sand relative to the bottom."""
        if self.base_plane is None:
            return np.zeros_like(current_frame)
        
        # Elevation = Base Distance - Current Distance
        # We use clip to ensure no negative height (below the box floor)
        elevation = np.clip(self.base_plane - current_frame, 0, 1000)
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

## BETA 2.0 - usage cancelled because it did not make a difference but i'm keeping it commented for documentation purposes
# import cv2
# import numpy as np

# class TerrainProcessor:
#     def __init__(self, base_plane_path=None):
#         self.base_plane = None  

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
#         elev = self.calculate_elevation(raw_frame)
#         elev_smoothed = cv2.GaussianBlur(elev, (5, 5), 0)
#         rgb_terrain = self.apply_color_map(elev_smoothed)
#         return rgb_terrain
    
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


## BETA 3 with filtering algorithms

import numpy as np
import cv2
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

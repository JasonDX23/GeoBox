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
        # Assume max sand height is 254
        max_h = 254.0
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
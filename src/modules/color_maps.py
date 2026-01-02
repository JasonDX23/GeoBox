import cv2
import numpy as np
import xml.etree.ElementTree as ET
import os

class ColorMapManager:
    def __init__(self):
        # 1. Initialize original OpenCV maps
        self.available_maps = {
            name.replace("COLORMAP_", "").capitalize(): getattr(cv2, name)
            for name in dir(cv2) if name.startswith("COLORMAP_")
        }
        self.sea_level_offset = 0.0
        
        # 2. Set hard fallbacks to prevent AttributeErrors
        self.current_map_id = cv2.COLORMAP_JET
        self.use_custom = False
        self.custom_lut = None

        # 3. Load the XML (This will update the default if successful)
        # Ensure this path matches your project structure exactly
        xml_path = "src/modules/hypsometric_tints.xml" 
        self.load_custom_xml(xml_path)

    def set_sea_level(self, value):
        """Sets the sea level offset in mm."""
        self.sea_level_offset = float(value)

    def load_custom_xml(self, path):
        if not os.path.exists(path):
            return
            
        tree = ET.parse(path)
        root = tree.getroot()
        heights = []
        colors = []
        
        for step in root.findall('step'):
            heights.append(float(step.get('height')))
            # Store as BGR for OpenCV compatibility
            colors.append([int(step.get('b')), int(step.get('g')), int(step.get('r'))])

        # Define the "Working Window" of your sandbox
        # This range should encompass your lowest and highest XML keys
        min_h, max_h = -250, 250 
        lut_range = np.linspace(min_h, max_h, 256)
        
        lut = np.zeros((256, 3), dtype=np.uint8)
        for i in range(3): # For B, G, R channels
            fp = [c[i] for c in colors]
            # Interpolate the XML colors across the 256-step LUT
            lut[:, i] = np.interp(lut_range, heights, fp)
        
        self.custom_lut = lut.reshape((256, 1, 3)).astype(np.uint8)
        self.available_maps[root.get('name', 'Custom')] = "CUSTOM"        
    def get_names(self):
        return list(self.available_maps.keys())

    def set_map_by_name(self, name):
        """Switch between OpenCV presets and Custom LUT."""
        val = self.available_maps.get(name, cv2.COLORMAP_JET)
        if val == "CUSTOM":
            self.use_custom = True
        else:
            self.use_custom = False
            self.current_map_id = val

    def apply(self, elevation_8bit):
        """Applies the color logic to the 8-bit depth frame."""
        if self.use_custom and self.custom_lut is not None:
            # cv2.LUT requires a 3-channel input to apply a 3-channel LUT
            # We merge the grayscale frame into 3 channels first
            merge_img = cv2.merge([elevation_8bit, elevation_8bit, elevation_8bit])
            return cv2.LUT(merge_img, self.custom_lut)
        else:
            return cv2.applyColorMap(elevation_8bit, self.current_map_id)

## Code to include slider to change colours to different heights - does not work well yet    
    # def __init__(self):
    #     # Elevation stops: (Elevation Value 0-255, BGR Color)
    #     self.stops = [
    #         {"val": 0,   "color": [128, 0, 0]},   # Deep Water
    #         {"val": 40,  "color": [255, 0, 0]},   # Shallow Water
    #         {"val": 60,  "color": [0, 255, 255]}, # Sand
    #         {"val": 150, "color": [0, 128, 0]},   # Grass
    #         {"val": 255, "color": [255, 255, 255]}# Peaks
    #     ]
    #     self.lut = self._rebuild_lut()

    # def _rebuild_lut(self):
    #     """Creates a 256x1 3-channel BGR LUT based on current stops."""
    #     lut = np.zeros((256, 1, 3), dtype=np.uint8)
    #     # Ensure stops are sorted by elevation value
    #     sorted_stops = sorted(self.stops, key=lambda x: x['val'])
        
    #     for i in range(len(sorted_stops) - 1):
    #         s1, s2 = sorted_stops[i], sorted_stops[i+1]
    #         idx1, idx2 = s1['val'], s2['val']
    #         # Linear interpolation between two colors
    #         for c in range(3): # B, G, R channels
    #             lut[idx1:idx2, 0, c] = np.linspace(s1['color'][c], s2['color'][c], idx2 - idx1)
    #     return lut

    # def update_stop_value(self, index, new_val):
    #     """Updates a specific stop's height and refreshes the LUT."""
    #     self.stops[index]['val'] = np.clip(new_val, 0, 255)
    #     self.lut = self._rebuild_lut()

    # def apply(self, elevation_8bit):
    #     # Using cv2.LUT is extremely efficient for custom mapping
    #     return cv2.LUT(cv2.merge([elevation_8bit]*3), self.lut)
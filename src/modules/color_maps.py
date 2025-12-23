import cv2
import numpy as np

class ColorMapManager:
    ### WORKING BASE CODE ###
    def __init__(self):
        # Dynamically find all COLORMAP_ constants in the cv2 library
        self.available_maps = {
            name.replace("COLORMAP_", "").capitalize(): getattr(cv2, name)
            for name in dir(cv2) if name.startswith("COLORMAP_")
        }
        # Set a default starting map
        self.current_map_id = cv2.COLORMAP_JET

    def get_names(self):
        """Returns a list of human-readable names for the GUI combo box."""
        return list(self.available_maps.keys())

    def set_map_by_name(self, name):
        """Updates the internal ID based on the selection name."""
        if name in self.available_maps:
            self.current_map_id = self.available_maps[name]

    def apply(self, elevation_8bit):
        """
        Applies the selected OpenCV colormap.
        Input must be an 8-bit single-channel image (0-255).
        """
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
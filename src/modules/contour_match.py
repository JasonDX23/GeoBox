import numpy as np
import cv2

class ContourMatchManager:
    def __init__(self, resolution=(480, 640)): # TODO: Is the resolution correct?
        self.resolution = resolution
        self.target_dem = None
        self.is_matching_mode = False

    def save_current_sand_as_dem(self, depth_frame, filename="src/modules/ContourMatch/dems/custom_terrain.dem"):
        "Captures current sand state and saves it as a Numpy binary file"
        dem_data = depth_frame.astype(np.float32)
        np.save(filename, dem_data)
        print(f"DEM recorded and saved to {filename}")

    def load_dem(self, filename):
        "Loads a pre-recorded DEM file into the matching engine"
        try:
            self.target_dem = np.load(filename)
            self.is_matching_mode = True
            print(f"Target DEM {filename} loaded successfully.")
        except FileNotFoundError:
            print("Error: DEM file not found.")

    def calculate_matching_guide(self, live_depth):
        "Computes the difference map and a quantitative matching score"
        if self.target_dem is None:
            return None, 0.0
        
        # Postive means sand is too high (remove), Negative means too low (add)
        diff = live_depth.astype(np.float32) - self.target_dem

        # Calculate RMSE (Root Mean Square Error) for a matching score #TODO: is this needed?
        score = np.sqrt(np.mean(diff**2))

        # Visual heatmap: Red (high), Blue (low), Green (correct)
        norm_diff = cv2.normalize(diff, None, 0,255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)

        return heatmap, score
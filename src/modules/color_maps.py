import numpy as np
import cv2
import matplotlib.pyplot as plt

class CPTLoader:
    """
    Parses both Segment-based (GMT) and Point-based CPT files.
    """
    @staticmethod
    def load_cpt(filepath, total_bins=256):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"[CPT] Error: File {filepath} not found.")
            return None

        # 1. Parse lines into a list of (z, r, g, b) tuples
        stops = []
        for line in lines:
            # Cleanup
            line = line.strip()
            if not line or line.startswith(('#', 'B', 'F', 'N')):
                continue
            
            tokens = line.split()
            
            try:
                # FORMAT A: Point-based (z R G B) - The format you provided
                if len(tokens) == 4:
                    z = float(tokens[0])
                    r, g, b = float(tokens[1]), float(tokens[2]), float(tokens[3])
                    stops.append((z, np.array([b, g, r]))) # Store as BGR for OpenCV
                
                # FORMAT B: Segment-based (z0 R0 G0 B0 z1 R1 G1 B1)
                elif len(tokens) >= 8:
                    # We only take the start point of each segment
                    z = float(tokens[0])
                    r, g, b = float(tokens[1]), float(tokens[2]), float(tokens[3])
                    stops.append((z, np.array([b, g, r])))
                    
                    # Also add the final endpoint of the last segment
                    # (This is a simplification, but effective for continuous gradients)
            except ValueError:
                continue

        if not stops:
            print("[CPT] Error: No valid data points found in file.")
            return None

        # 2. Sort stops by Z (Depth)
        stops.sort(key=lambda x: x[0])

        # 3. Generate Lookup Table (LUT)
        min_z = stops[0][0]
        max_z = stops[-1][0]
        total_range = max_z - min_z
        
        # Avoid division by zero
        if total_range == 0: total_range = 1.0

        lut = np.zeros((total_bins, 1, 3), dtype=np.uint8)

        for i in range(total_bins):
            # Calculate the Z value this index represents
            current_z = min_z + (i / (total_bins - 1)) * total_range
            
            # Find the two stops this Z falls between
            color = stops[0][1] # Default to first
            
            for j in range(len(stops) - 1):
                z_start, c_start = stops[j]
                z_end, c_end = stops[j+1]
                
                if z_start <= current_z <= z_end:
                    # Interpolate
                    fraction = (current_z - z_start) / (z_end - z_start)
                    color = (1 - fraction) * c_start + fraction * c_end
                    break
                elif current_z > stops[-1][0]:
                    color = stops[-1][1] # Clamp top
            
            lut[i, 0] = color.astype(np.uint8)

        print(f"[CPT] Loaded {len(stops)} color stops. Range: {min_z} to {max_z}.")
        return lut


class HybridColorMapper:
    def __init__(self):
        self.mode = 'preset' 
        self.preset_name = 'terrain' 
        self.custom_stops = []
        
        # Initialize LUT storage (256, 1, 3) 
        self.lut = np.zeros((256, 1, 3), dtype=np.uint8)
        self.cpt_loader = CPTLoader()
        
        self.rebuild_lut()

    def set_mode_preset(self, name):
        self.mode = 'preset'
        self.preset_name = name
        self.rebuild_lut()

    def set_mode_custom(self, stops):
        self.mode = 'custom'
        self.custom_stops = sorted(stops, key=lambda x: x[0])
        self.rebuild_lut()

    def load_cpt_file(self, path):
        new_lut = self.cpt_loader.load_cpt(path)
        if new_lut is not None:
            self.lut = new_lut
            self.mode = 'file'
        else:
            print("[Color] Failed to load CPT. Reverting to previous state.")

    def rebuild_lut(self):
        if self.mode == 'preset':
            try:
                cmap = plt.get_cmap(self.preset_name)
                gradient = cmap(np.arange(256))[:, :3] 
                # Convert RGB to BGR
                lut_data = (gradient * 255).astype(np.uint8)[:, ::-1]
                self.lut = lut_data.reshape(256, 1, 3)
            except Exception as e:
                print(f"[Color] Error loading preset '{self.preset_name}': {e}")
                self.lut = np.dstack([np.arange(256)]*3).astype(np.uint8).reshape(256, 1, 3)
            
        elif self.mode == 'custom':
            if not self.custom_stops: return
            
            keys = [int(s[0] * 255) for s in self.custom_stops]
            b_vals = [s[1][0] for s in self.custom_stops]
            g_vals = [s[1][1] for s in self.custom_stops]
            r_vals = [s[1][2] for s in self.custom_stops]
            
            x = np.arange(256)
            self.lut[:, 0, 0] = np.interp(x, keys, b_vals).astype(np.uint8)
            self.lut[:, 0, 1] = np.interp(x, keys, g_vals).astype(np.uint8)
            self.lut[:, 0, 2] = np.interp(x, keys, r_vals).astype(np.uint8)

    def apply(self, height_map, sea_offset=0.0):
        """
        Applies the color map.
        """
        # 1. Normalize and Apply Offset
        indices_float = (height_map * 255.0)
        if sea_offset != 0.0:
            indices_float -= (sea_offset * 255.0)
            
        # 2. Create 1-channel index map
        indices = np.clip(indices_float, 0, 255).astype(np.uint8)
        
        # 3. EXPAND TO 3 CHANNELS (Crucial for cv2.LUT)
        indices_bgr = cv2.cvtColor(indices, cv2.COLOR_GRAY2BGR)
        
        # 4. Apply LUT
        return cv2.LUT(indices_bgr, self.lut)
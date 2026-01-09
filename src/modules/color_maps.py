import numpy as np
import matplotlib.pyplot as plt

class HybridColorMapper:
    def __init__(self):
        self.mode = 'preset' # or 'custom'
        self.preset_name = 'terrain'
        self.custom_stops = []
        self.lut = np.zeros((256, 3), dtype=np.uint8)
        self.rebuild_lut()

    def set_mode_preset(self, name):
        self.mode = 'preset'
        self.preset_name = name
        self.rebuild_lut()

    def set_mode_custom(self, stops):
        self.mode = 'custom'
        self.custom_stops = sorted(stops, key=lambda x: x[0])
        self.rebuild_lut()

    def rebuild_lut(self):
        if self.mode == 'preset':
            cmap = plt.get_cmap(self.preset_name)
            gradient = cmap(np.arange(256))[:, :3]
            self.lut = (gradient * 255).astype(np.uint8)[:, ::-1] # RGB->BGR
            
        elif self.mode == 'custom':
            if not self.custom_stops: return
            
            keys = [int(s[0] * 255) for s in self.custom_stops]
            b_vals = [s[1][0] for s in self.custom_stops]
            g_vals = [s[1][1] for s in self.custom_stops]
            r_vals = [s[1][2] for s in self.custom_stops]
            
            x = np.arange(256)
            self.lut[:, 0] = np.interp(x, keys, b_vals)
            self.lut[:, 1] = np.interp(x, keys, g_vals)
            self.lut[:, 2] = np.interp(x, keys, r_vals)

    def apply(self, height_map, sea_offset=0.0):
        # Apply sea level offset to indices
        indices = (height_map * 255).astype(np.float32)
        indices = indices - (sea_offset * 255)
        indices = np.clip(indices, 0, 255).astype(np.uint8)
        
        return self.lut[indices]
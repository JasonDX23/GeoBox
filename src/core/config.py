import json
import numpy as np
import os

CONFIG_FILE = "assets/config/calibration.json"

class ConfigManager:
    def __init__(self):
        # Default 2.5D Matrix (Pass-through)
        # x = 1*u + 0*v + 0*d + 0
        # y = 0*u + 1*v + 0*d + 0
        self.default_matrix = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ]
        
        self.default_config = {
            "roi": [0, 0, 640, 480],
            "matrix": self.default_matrix, # Renamed from 'homography'
            "depth_range": [600, 1100]
        }
        self.data = self.load()

    def load(self):
        if not os.path.exists(CONFIG_FILE): return self.default_config
        try:
            with open(CONFIG_FILE, 'r') as f:
                d = json.load(f)
                if "matrix" not in d: d["matrix"] = self.default_config["matrix"]
                return d
        except: return self.default_config

    def save(self, roi, matrix, depth_range):
        # Ensure matrix is list
        m_list = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        data = {"roi": roi, "matrix": m_list, "depth_range": depth_range}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
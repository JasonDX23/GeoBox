import json
import numpy as np
import os

CONFIG_FILE = "assets/config/calibration.json"

class ConfigManager:
    def __init__(self):
        # Default Quadratic Matrix (Identity Linear)
        # [u, v, d, u2, v2, uv, 1]
        self.default_matrix = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], # x = u
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # y = v
        ]
        
        self.default_config = {
            "roi": [0, 0, 640, 480],
            "matrix": self.default_matrix,
            "depth_range": [600, 1100]
        }
        self.data = self.load()

    def load(self):
        if not os.path.exists(CONFIG_FILE): return self.default_config
        try:
            with open(CONFIG_FILE, 'r') as f:
                d = json.load(f)
                return d
        except: return self.default_config

    def save(self, roi, matrix, depth_range):
        m_list = matrix.tolist() if hasattr(matrix, "tolist") else matrix
        data = {"roi": roi, "matrix": m_list, "depth_range": depth_range}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
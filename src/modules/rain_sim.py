import numpy as np

class RainSimulator:
    def __init__(self, shape):
        self.water_layer = np.zeros(shape, dtype=np.float32)
        self.evaporation_rate = 0.02

    def update(self, terrain_map):
        """
        Simulates flow. Water moves to lower adjacent cells.
        terrain_map: Normalized height map (0.0 - 1.0)
        """
        # Spawn rain randomly
        if np.random.rand() > 0.90:
            rx = np.random.randint(0, terrain_map.shape[1])
            ry = np.random.randint(0, terrain_map.shape[0])
            self.water_layer[ry, rx] += 1.0

        # Calculate gradient (slope)
        dy, dx = np.gradient(terrain_map)
        
        # Simple flow: shift water indices based on negative gradient (downhill)
        # Vectorized aproximation of flow
        flow_y = np.roll(self.water_layer, 1, axis=0) * (dy > 0.01) # Flow Down
        flow_x = np.roll(self.water_layer, 1, axis=1) * (dx > 0.01) # Flow Right
        
        # Decay/Evaporation
        self.water_layer = np.clip(self.water_layer + (flow_y + flow_x) * 0.1 - self.evaporation_rate, 0, 1)

        return self.water_layer
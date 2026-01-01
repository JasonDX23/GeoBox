import numpy as np

class WaterSim:
    def __init__(self, height=150, width=200, dx=1.0):
        self.height, self.width, self.dx = height, width, dx
        self.terrain = np.zeros((height, width), dtype=np.float32)
        self.water_h = np.zeros((height, width), dtype=np.float32)
        self.flux_x = np.zeros((height, width), dtype=np.float32)
        self.flux_y = np.zeros((height, width), dtype=np.float32)

    def update_simulation(self,dt, gravity=9.81, attenuation=0.99):
        global water_h, flux_x, flux_y

        surface = self.terrain + water_h
    
        # Calculate gradients (simplified finite difference)
        grad_x = (surface[:, 1:] - surface[:, :-1]) / self.dx
        grad_y = (surface[1:, :] - surface[:-1, :]) / self.dx
        
        self.flux_x[:, :-1] -= dt * gravity * (surface[:, 1:] - surface[:, :-1]) / self.dx
        self.flux_y[:-1, :] -= dt * gravity * (surface[1:, :] - surface[:-1, :]) / self.dx
        
        # Apply damping to simulate friction/viscosity
        self.flux_x *= attenuation
        self.flux_y *= attenuation
        
        # Divergence: Change in height based on flux in/out
        # dH/dt = - (dFluxX/dx + dFluxY/dy)
        outflow_x = np.diff(self.flux_x, prepend=0, axis=1)
        outflow_y = np.diff(self.flux_y, prepend=0, axis=0)
        
        self.water_h -= dt * (outflow_x + outflow_y) / self.dx
        self.water_h = np.maximum(0, self.water_h) # Prevent negative water
import numpy as np
import cv2

class FluidSimulator:
    def __init__(self, shape=(480, 640), physics_scale=0.25):
        """
        physics_scale: 0.25 means physics runs at 1/4 resolution (120x160)
        """
        self.orig_shape = shape
        self.h, self.w = int(shape[0] * physics_scale), int(shape[1] * physics_scale)
        
        # Physics State Buffers
        self.water_depth = np.zeros((self.h, self.w), dtype=np.float32)
        self.new_water = np.zeros_like(self.water_depth)
        self.flux_L = np.zeros_like(self.water_depth) # Left
        self.flux_R = np.zeros_like(self.water_depth) # Right
        self.flux_T = np.zeros_like(self.water_depth) # Top
        self.flux_B = np.zeros_like(self.water_depth) # Bottom
        
        # Parameters
        self.pipe_len = 1.0     # Virtual pipe length (distance between cells)
        self.cross_sect = 0.5   # Pipe cross-section area (flow capacity)
        self.gravity = 9.81
        self.dt = 0.05          # Simulation timestep
        
        self.evaporation_rate = 0.001
        self.rain_rate = 0.01
        self.is_raining = False

    def set_rain(self, enabled, rate):
        self.is_raining = enabled
        self.rain_rate = rate

    def step(self, terrain_hires):
        """
        Updates the fluid simulation.
        terrain_hires: High-res (480x640) 0.0-1.0 height map.
        """
        # 1. Downscale Terrain to Physics Grid
        terrain = cv2.resize(terrain_hires, (self.w, self.h), interpolation=cv2.INTER_AREA)
        
        # 2. Add Rain
        if self.is_raining:
            # Random droplets vs Global rain? SARndbox usually does global gentle rain
            # We'll add uniform rain for filling, plus random noise for variety
            rain_mask = np.random.rand(self.h, self.w) < 0.1
            self.water_depth[rain_mask] += self.rain_rate

        # 3. Calculate Total Height (Terrain + Water)
        H = terrain + self.water_depth

        # 4. Flux Calculation (Vectorized Virtual Pipes)
        # Calculate height difference between cell and neighbors
        # Note: We shift arrays to align neighbors
        
        # Delta H to Left: H(x,y) - H(x-1, y)
        d_L = H - np.roll(H, 1, axis=1); d_L[:, 0] = 0 # Boundary fix
        # Delta H to Right
        d_R = H - np.roll(H, -1, axis=1); d_R[:, -1] = 0
        # Delta H to Top
        d_T = H - np.roll(H, 1, axis=0); d_T[0, :] = 0
        # Delta H to Bottom
        d_B = H - np.roll(H, -1, axis=0); d_B[-1, :] = 0

        # Update Flux (Accelerate flow based on gravity difference)
        # Flux += dt * area * gravity * height_diff / pipe_len
        const = self.dt * self.cross_sect * self.gravity / self.pipe_len
        
        self.flux_L = np.maximum(0, self.flux_L + const * d_L)
        self.flux_R = np.maximum(0, self.flux_R + const * d_R)
        self.flux_T = np.maximum(0, self.flux_T + const * d_T)
        self.flux_B = np.maximum(0, self.flux_B + const * d_B)

        # 5. Scaling Flux to prevent volume loss (CFL condition approximation)
        # Total out-flux per cell
        out_flux = self.flux_L + self.flux_R + self.flux_T + self.flux_B
        
        # If out-flux > current water volume, scale it down
        # We process K factor: Volume / (OutFlux * dt)
        with np.errstate(divide='ignore', invalid='ignore'):
            K = np.minimum(1.0, self.water_depth * (self.w * self.h) / (out_flux * self.dt + 1e-6))
        
        # Apply scaling
        self.flux_L *= K
        self.flux_R *= K
        self.flux_T *= K
        self.flux_B *= K

        # 6. Water Volume Update
        # Change = Inflow - Outflow
        # Inflow comes from neighbors' Outflow
        inflow = (np.roll(self.flux_L, -1, axis=1) + 
                  np.roll(self.flux_R, 1, axis=1) + 
                  np.roll(self.flux_T, -1, axis=0) + 
                  np.roll(self.flux_B, 1, axis=0))
        
        outflow = (self.flux_L + self.flux_R + self.flux_T + self.flux_B)
        
        d_vol = self.dt * (inflow - outflow)
        self.water_depth += d_vol

        # 7. Evaporation & Cleanup
        self.water_depth -= self.evaporation_rate
        self.water_depth = np.maximum(0, self.water_depth)

        # 8. Render Mask (Upscale for display)
        # Returns float32 mask 0.0-1.0
        return cv2.resize(self.water_depth, (self.orig_shape[1], self.orig_shape[0]), interpolation=cv2.INTER_LINEAR)
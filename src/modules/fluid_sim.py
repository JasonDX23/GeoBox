import numpy as np
import cv2

class FluidSimulator:
    def __init__(self, shape=(480, 640), physics_scale=0.5):
        self.orig_shape = shape
        self.h, self.w = int(shape[0] * physics_scale), int(shape[1] * physics_scale)
        
        # Physics State
        self.water_depth = np.zeros((self.h, self.w), dtype=np.float32)
        self.flux_L = np.zeros_like(self.water_depth)
        self.flux_R = np.zeros_like(self.water_depth)
        self.flux_T = np.zeros_like(self.water_depth)
        self.flux_B = np.zeros_like(self.water_depth)
        
        # --- HIGH SPEED PARAMETERS ---
        self.pipe_len = 1.0     
        self.gravity = 9.81
        
        # 1. Pipe Capacity: Increased 1.2 -> 2.5 (Wide pipes = fast spreading)
        self.cross_sect = 2.0   
        
        # 2. Simulation Speed: Increased 8 -> 20 updates per frame
        # This makes the water travel much further per video frame.
        self.sub_steps = 30 
        
        # 3. Time Step: Reduced slightly for stability (since steps are higher)
        self.dt = 0.01        
        
        # 4. Low Friction: 0.999 keeps momentum high (splashy)
        self.damping = 0.999 
        
        # Environment
        self.evaporation_rate = 0.0005
        self.rain_rate = 0.02
        self.is_raining = False

        # Interaction
        self.cursor_active = False
        self.cursor_x = 0
        self.cursor_y = 0
        self.source_radius = 5  
        self.source_rate = 0.4   # Increased pour rate to match flow speed

        # Visuals
        self.water_color = np.array([255, 120, 20], dtype=np.float32) 
        self.foam_color = np.array([255, 255, 255], dtype=np.float32)

    def set_rain(self, enabled, rate):
        self.is_raining = enabled
        self.rain_rate = rate

    def set_interaction_point(self, x, y, active):
        self.cursor_x = x
        self.cursor_y = y
        self.cursor_active = active

    def get_water_visual(self, depth_mask):
        """
        Fixed: Forces Electric Blue color even for shallow/fast water.
        """
        # 0. Safety
        depth_mask = np.nan_to_num(depth_mask, nan=0.0)
        
        # 1. Upscale & Smooth
        hires_depth = cv2.resize(depth_mask, (self.orig_shape[1], self.orig_shape[0]), interpolation=cv2.INTER_CUBIC)
        hires_depth = cv2.GaussianBlur(hires_depth, (9, 9), 1.5)
        
        mask = hires_depth > 0.005 # Lower threshold to catch thin water
        if not np.any(mask):
            return np.zeros((self.orig_shape[0], self.orig_shape[1], 3), dtype=np.uint8)

        # 2. COLOR GRADIENT (The Fix)
        # Previously: * 2.5 (Required 40cm depth for full blue)
        # Now: * 20.0 (Only requires 5cm depth for full blue)
        # This ensures fast-moving thin water still looks deep and electric.
        d_norm = np.clip(hires_depth * 20.0, 0, 1)[:, :, np.newaxis]
        
        # Colors (BGR)
        # Shallow: Light Blue (instead of white/cyan)
        c_shallow = np.array([255, 180, 50], dtype=np.float32) 
        # Deep: Electric Blue (The target look)
        c_deep    = np.array([255, 100, 0], dtype=np.float32)   
        
        base_color = (c_shallow * (1 - d_norm)) + (c_deep * d_norm)

        # 3. Turbulence (White Highlights)
        turbulence = cv2.Laplacian(hires_depth, cv2.CV_32F)
        turbulence = np.abs(turbulence) * 80.0 # Increased contrast
        turbulence = np.clip(turbulence, 0, 1)[:, :, np.newaxis]

        # Composite
        final_color = base_color + (self.foam_color * turbulence)
        final_color = np.clip(final_color, 0, 255).astype(np.uint8)
        
        # 4. Bloom (Glow)
        glow = cv2.GaussianBlur(final_color, (21, 21), 0)
        final_color = cv2.addWeighted(final_color, 0.7, glow, 0.6, 0)
        
        visual = np.where(mask[:,:,None], final_color, np.zeros_like(final_color))
        return visual

    def step(self, terrain_hires):
        terrain = cv2.resize(terrain_hires, (self.w, self.h), interpolation=cv2.INTER_AREA)
        
        # 1. Cursor Interaction
        if self.cursor_active:
            px = int(self.cursor_x * (self.w / self.orig_shape[1]))
            py = int(self.cursor_y * (self.h / self.orig_shape[0]))
            pr = int(self.source_radius * (self.w / self.orig_shape[1]))
            
            y1 = max(0, py - pr); y2 = min(self.h, py + pr)
            x1 = max(0, px - pr); x2 = min(self.w, px + pr)
            
            if x2 > x1 and y2 > y1:
                self.water_depth[y1:y2, x1:x2] += self.source_rate

        # 2. Global Rain
        if self.is_raining:
            drops = np.random.rand(self.h, self.w) < 0.05
            self.water_depth[drops] += self.rain_rate

        # 3. High-Speed Physics Loop
        for _ in range(self.sub_steps):
            H = terrain + self.water_depth
            
            # Vectorized neighbor lookup
            d_L = H - np.roll(H, 1, axis=1); d_L[:, 0] = 0 
            d_R = H - np.roll(H, -1, axis=1); d_R[:, -1] = 0
            d_T = H - np.roll(H, 1, axis=0); d_T[0, :] = 0
            d_B = H - np.roll(H, -1, axis=0); d_B[-1, :] = 0

            # Update Flux
            const = self.dt * self.cross_sect * self.gravity / self.pipe_len
            self.flux_L = np.maximum(0, self.flux_L + const * d_L)
            self.flux_R = np.maximum(0, self.flux_R + const * d_R)
            self.flux_T = np.maximum(0, self.flux_T + const * d_T)
            self.flux_B = np.maximum(0, self.flux_B + const * d_B)

            # Stability (CFL) Check
            out_flux = self.flux_L + self.flux_R + self.flux_T + self.flux_B
            denom = out_flux * self.dt + 1e-6
            
            # Safety brake: K cannot exceed 1.0
            K = np.minimum(1.0, self.water_depth / denom) * self.damping
            
            self.flux_L *= K; self.flux_R *= K; self.flux_T *= K; self.flux_B *= K

            # Volume Update
            inflow = (np.roll(self.flux_L, -1, axis=1) + np.roll(self.flux_R, 1, axis=1) + 
                      np.roll(self.flux_T, -1, axis=0) + np.roll(self.flux_B, 1, axis=0))
            outflow = (self.flux_L + self.flux_R + self.flux_T + self.flux_B)
            
            self.water_depth += self.dt * (inflow - outflow)
            
            # Clamping
            self.water_depth = np.maximum(0, self.water_depth)
            self.water_depth = np.minimum(self.water_depth, 10.0)
            self.water_depth[:, 0] = 0; self.water_depth[:, -1] = 0
            self.water_depth[0, :] = 0; self.water_depth[-1, :] = 0

        self.water_depth = np.maximum(0, self.water_depth - self.evaporation_rate)
        return self.get_water_visual(self.water_depth)
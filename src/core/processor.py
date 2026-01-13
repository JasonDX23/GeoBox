import numpy as np
import cv2

class DepthProcessor:
    def __init__(self, min_depth=600, max_depth=1100):
        self.min_d = min_depth
        self.max_d = max_depth
        self.accumulated = None 
        
        # INCREASED SPEED: 0.3 means "30% New Data, 70% Old Data"
        # This removes the "Ghost Hand" trail much faster.
        self.history_alpha = 0.3 # TEST 
    
    def auto_range(self, depth_frame):
        safe_depth = np.nan_to_num(depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
        valid_pixels = safe_depth[(safe_depth > 400) & (safe_depth < 1500)]
        
        if len(valid_pixels) == 0:
            return 800, 900 
            
        p10 = np.percentile(valid_pixels, 10)
        p90 = np.percentile(valid_pixels, 90)
        
        mid = (p10 + p90) / 2
        spread = p90 - p10
        if spread < 30: spread = 30
        
        new_min = int(mid - (spread / 2) - 10)
        new_max = int(mid + (spread / 2) + 10)
        
        return new_min, new_max

    def _prepare_frames(self, depth_frame):
        """Shared preprocessing for both Visuals and Physics"""
        # 0. NaN Guard
        depth_frame = np.nan_to_num(depth_frame, nan=0.0)
        
        # 1. Blur (Remove Quantization Lines)
        clean_depth = cv2.GaussianBlur(depth_frame, (11, 11), 0)

        # 2. Temporal Smoothing
        if self.accumulated is None:
            self.accumulated = clean_depth
        else:
            cv2.accumulateWeighted(clean_depth, self.accumulated, self.history_alpha)
        
        return self.accumulated

    def normalize(self, depth_frame):
        """Generates the VISUAL terrain (Includes Hands)"""
        filtered = self._prepare_frames(depth_frame)
        
        # Clip & Normalize
        clipped = np.clip(filtered, self.min_d, self.max_d)
        range_diff = max(1, self.max_d - self.min_d)
        norm = (self.max_d - clipped) / range_diff
        
        norm = np.clip(norm, 0.0, 1.0)
        norm = np.power(norm, 1.2) # Gamma
        
        return norm

    def normalize_for_physics(self, depth_frame):
        """
        Generates the PHYSICS terrain (Removes Hands).
        This ensures water falls THROUGH the hand, not OFF it.
        """
        # We reuse the smoothed frame from _prepare_frames (accessed via self.accumulated)
        # But for physics, we don't want to smooth the 'hand removal' too much, 
        # so we operate on the current frame or the accumulated one.
        # Let's use accumulated to keep it synced.
        if self.accumulated is None: return self.normalize(depth_frame)
        
        phys_depth = self.accumulated.copy()
        
        # 1. Detect Hands: Pixels closer than the 'min_depth' (Sand Peak)
        # We add a buffer (e.g. 50mm) to distinguish 'High Sand' from 'Hand'
        hand_threshold = self.min_d - 30 
        is_hand = (phys_depth > 100) & (phys_depth < hand_threshold)
        
        # 2. Erase Hands: Replace hand pixels with the "Average Sand Depth"
        # This flattens the hand so it acts like open air/flat sand to the water.
        # We find the median of the NON-HAND area to use as the fill value.
        valid_sand = phys_depth[(phys_depth >= hand_threshold) & (phys_depth < self.max_d)]
        
        if len(valid_sand) > 0:
            fill_val = np.median(valid_sand)
        else:
            fill_val = self.max_d # Default to floor if no sand seen
            
        phys_depth[is_hand] = fill_val
        
        # 3. Standard Normalization on the "Erased" map
        clipped = np.clip(phys_depth, self.min_d, self.max_d)
        range_diff = max(1, self.max_d - self.min_d)
        norm = (self.max_d - clipped) / range_diff
        
        return np.clip(norm, 0.0, 1.0)

    def get_contour_layer(self, height_map, interval=0.05):
        smooth_map = cv2.GaussianBlur(height_map, (15, 15), 0)
        h, w = height_map.shape
        big_h = cv2.resize(smooth_map, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        viz = np.zeros((h*2, w*2, 4), dtype=np.uint8)
        
        for level in np.arange(interval, 0.95, interval):
            thresh = np.zeros_like(big_h, dtype=np.uint8)
            thresh[big_h > level] = 255
            cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, cnts, -1, (0,0,0,255), 4) 
            
        return cv2.resize(viz, (w, h), interpolation=cv2.INTER_AREA)
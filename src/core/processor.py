import numpy as np
import cv2

class DepthProcessor:
    def __init__(self, min_depth=254, max_depth=1100):
        self.min_d = min_depth
        self.max_d = max_depth
        
        # --- C++ PORT SETTINGS ---
        # Corresponds to 'numAveragingSlots' in C++
        # Higher = Smoother but more latency. 10-15 is a good balance.
        self.history_len = 10 
        
        # Corresponds to 'maxVariance' in C++. 
        # If a pixel varies more than this (std_dev), we define it as 'unstable'
        self.stability_threshold = 8.0 
        
        # Corresponds to 'hysteresis'. 
        # Even if stable, only update if value changes by at least this much.
        self.hysteresis = 2.0
        
        # The Ring Buffer: Stores the last N frames of raw depth
        self.frame_buffer = None
        self.buffer_index = 0
        
        # The "Valid Buffer" (The last known good stable image)
        self.last_stable_frame = None

    def auto_range(self, depth_frame):
        # [Same as your original code]
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

    def _process_statistical_filter(self, depth_frame):
        """
        Implements the C++ FrameFilter logic:
        1. Store frame in ring buffer.
        2. Calculate Mean and StdDev across the time axis.
        3. Update output ONLY if pixel is stable (Low StdDev).
        """
        h, w = depth_frame.shape
        
        # 1. Initialize Buffers if first run
        if self.frame_buffer is None or self.frame_buffer.shape[1:] != (h, w):
            self.frame_buffer = np.zeros((self.history_len, h, w), dtype=np.float32)
            self.last_stable_frame = np.copy(depth_frame).astype(np.float32)

        # 2. Add new frame to Ring Buffer (Replaces oldest frame)
        # NaN handling: Replace NaNs with 0 or last known good value
        clean_frame = np.nan_to_num(depth_frame, nan=0.0)
        self.frame_buffer[self.buffer_index] = clean_frame
        
        # Increment/Loop index
        self.buffer_index = (self.buffer_index + 1) % self.history_len

        # 3. Calculate Statistics (Vectorized)
        # We calculate the Standard Deviation and Mean along axis 0 (Time)
        # This replaces the 'statBuffer' and 'averagingBuffer' loops in C++
        buffer_mean = np.mean(self.frame_buffer, axis=0)
        buffer_std = np.std(self.frame_buffer, axis=0)

        # 4. Stability Check (The Magic Step)
        # C++: if(variance <= maxVariance)
        is_stable = buffer_std < self.stability_threshold
        
        # 5. Hysteresis Check
        # C++: if(abs(newFiltered - oldVal) >= hysteresis)
        diff = np.abs(buffer_mean - self.last_stable_frame)
        significant_change = diff > self.hysteresis

        # 6. Determine Update Mask
        # We update pixels that are STABLE AND have CHANGED SIGNIFICANTLY
        update_mask = is_stable & significant_change

        # 7. Update the Result
        # Where update_mask is True, use new Mean.
        # Where update_mask is False, keep self.last_stable_frame (Retain Valids)
        self.last_stable_frame[update_mask] = buffer_mean[update_mask]
        
        # 8. Spatial Filter (Low-pass filter from C++)
        # C++ performs a custom neighbor averaging. GaussianBlur is the OpenCV equivalent.
        # We perform this on the STABLE result.
        final_output = cv2.GaussianBlur(self.last_stable_frame, (5, 5), 0)
        
        return final_output

    def normalize(self, depth_frame):
        """Generates the VISUAL terrain (Includes Hands)"""
        # Run the statistical filter instead of simple smooth
        filtered = self._process_statistical_filter(depth_frame)
        
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
        """
        # Use the stable frame we calculated in normalize()
        # If normalize() wasn't called yet, run the filter.
        if self.last_stable_frame is None:
            phys_depth = self._process_statistical_filter(depth_frame)
        else:
            phys_depth = self.last_stable_frame.copy()
        
        # 1. Detect Hands: Pixels closer than the 'min_depth' (Sand Peak)
        hand_threshold = self.min_d - 30 
        is_hand = (phys_depth > 100) & (phys_depth < hand_threshold)
        
        # 2. Erase Hands
        valid_sand = phys_depth[(phys_depth >= hand_threshold) & (phys_depth < self.max_d)]
        
        if len(valid_sand) > 0:
            fill_val = np.median(valid_sand)
        else:
            fill_val = self.max_d 
            
        phys_depth[is_hand] = fill_val
        
        # 3. Normalize
        clipped = np.clip(phys_depth, self.min_d, self.max_d)
        range_diff = max(1, self.max_d - self.min_d)
        norm = (self.max_d - clipped) / range_diff
        
        return np.clip(norm, 0.0, 1.0)

    def get_contour_layer(self, height_map, interval=0.10):
        # [Same as your original code]
        smooth_map = cv2.GaussianBlur(height_map, (15, 15), 0)
        h, w = height_map.shape
        big_h = cv2.resize(smooth_map, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        
        viz = np.zeros((h*2, w*2, 4), dtype=np.uint8)
        
        for level in np.arange(interval, 0.95, interval):
            thresh = np.zeros_like(big_h, dtype=np.uint8)
            thresh[big_h > level] = 255
            cnts, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(viz, cnts, -1, (255,255,255,255), 2)
            
        return cv2.resize(viz, (w, h), interpolation=cv2.INTER_AREA)
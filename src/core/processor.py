import cv2
import numpy as np

class DepthProcessor:
    def __init__(self, min_depth=600, max_depth=1100):
        self.min_d = min_depth
        self.max_d = max_depth
        
        # Temporal Smoothing State
        self.accumulated_depth = None
        self.alpha = 0.3 # Lower = smoother but more "ghosting" lag
        
        # Shadow Filling State
        self.dilate_kernel = np.ones((3,3), np.uint8)

    def normalize(self, raw_depth):
        """
        Full pipeline: Temporal Smooth -> Shadow Fill -> Normalize
        """
        # 1. Temporal Smoothing (Exponential Moving Average)
        # Convert to float for accumulation
        if self.accumulated_depth is None:
            self.accumulated_depth = raw_depth.astype(np.float32)
        else:
            cv2.accumulateWeighted(raw_depth.astype(np.float32), 
                                 self.accumulated_depth, self.alpha)
        
        smoothed = self.accumulated_depth.astype(np.uint16)

        # 2. Advanced Shadow Filling (Iterative Dilation)
        # Shadows are 0 (or 2047 in some Kinect modes). Assuming 0 here.
        filled = smoothed.copy()
        mask = (filled == 0).astype(np.uint8)
        
        # Grow valid pixels into the mask 3 times
        # This closes gaps caused by parallax without destroying details elsewhere
        for _ in range(3):
            if np.count_nonzero(mask) == 0: break
            # Dilate the image (spreads high values)
            dilated = cv2.dilate(filled, self.dilate_kernel)
            # Only fill where the original was 0
            filled[mask == 1] = dilated[mask == 1]
            # Update mask
            mask = (filled == 0).astype(np.uint8)

        # 3. Clip and Normalize
        clipped = np.clip(filled, self.min_d, self.max_d)
        norm = (self.max_d - clipped) / (self.max_d - self.min_d)
        
        return norm

    def get_contour_layer(self, height_map, interval=0.05):
        h, w = height_map.shape
        overlay = np.zeros((h, w, 4), dtype=np.uint8)
        if interval <= 0.01: return overlay

        # 4. Spatial Smoothing for Contours
        # Gaussian Blur reduces "jagged" lines caused by pixel steps
        smooth_map = cv2.GaussianBlur(height_map, (5, 5), 0)
        
        levels = np.arange(interval, 1.0, interval)
        img_8u = (smooth_map * 255).astype(np.uint8)
        
        for level in levels:
            thresh_val = int(level * 255)
            ret, bin_img = cv2.threshold(img_8u, thresh_val, 255, cv2.THRESH_BINARY)
            
            # Find contours on the binary mask
            contours, _ = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw with Anti-Aliasing
            cv2.drawContours(overlay, contours, -1, (0, 0, 0, 255), 1, cv2.LINE_AA)
            
        return overlay
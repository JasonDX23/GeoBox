import cv2
import numpy as np
import traceback

class DEMHandler:
    """
    Handles loading Digital Elevation Models (DEMs) and computing 
    guidance visualizations (difference between sand and target).
    """
    def __init__(self, target_width=640, target_height=480):
        self.target_w = target_width
        self.target_h = target_height
        self.active = False
        
        # The reference heightmap (float32, normalized or in mm)
        self.dem_grid = None
        self.dem_loaded = False
        
        # Tolerance for "Green" zone (in millimeters if using physics depth)
        self.tolerance = 15.0 

    def load_dem(self, file_path, min_depth_mm, max_depth_mm):
        """
        Loads a DEM from .npy (native) or images (tiff/png/jpg).
        """
        try:
            # --- CASE 1: NumPy Binary (Recommended for Save/Load) ---
            if file_path.endswith('.npy'):
                data = np.load(file_path)
                # .npy stores exact millimeters, so we just check shapes
                if data.shape != (self.target_h, self.target_w):
                     # Resize if dimensions don't match (e.g. from different resolution)
                     data = cv2.resize(data, (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
                
                self.dem_grid = data
                self.dem_loaded = True
                self.active = True
                print(f"[DEM] Loaded raw physics data from {file_path}")
                return True

            # --- CASE 2: Image Files (TIFF/PNG) ---
            # Use IMREAD_UNCHANGED to try and get 16-bit/32-bit data
            img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            
            if img is None:
                print(f"[DEM] Error: OpenCV failed to decode {file_path}")
                return False

            # ... (Rest of your existing image normalization logic) ...
            # Handle multi-channel images
            if len(img.shape) > 2:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_resized = cv2.resize(img, (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)

            # Normalize based on type
            if img_resized.dtype == np.uint8:
                img_norm = img_resized.astype(np.float32) / 255.0
            elif img_resized.dtype == np.uint16:
                img_norm = img_resized.astype(np.float32) / 65535.0
            elif img_resized.dtype == np.float32:
                # If OpenCV actually manages to read a float TIFF, use it directly
                # assuming it is already in meters or normalized? 
                # Ideally we assume normalized 0-1 for generic images, 
                # but if it matches your depth range, you might need logic here.
                # For safety, let's assume float TIFFs are 0.0-1.0 normalized unless huge values detected
                if np.max(img_resized) > 1.0:
                     # Looks like raw mm or meters? Treat as raw MM if values are large
                     self.dem_grid = img_resized
                     self.dem_loaded = True
                     self.active = True
                     return True
                else:
                     img_norm = img_resized
            else:
                img_norm = img_resized.astype(np.float32) / np.max(img_resized)

            self.dem_grid = min_depth_mm + (img_norm * (max_depth_mm - min_depth_mm))
            self.dem_loaded = True
            self.active = True
            print(f"[DEM] Loaded image {file_path}")
            return True

        except Exception as e:
            print(f"[DEM] Exception loading file: {e}")
            traceback.print_exc()
            return False

    def save_dem(self, file_path, current_depth_mm):
        """
        Saves as .npy (Perfect Data) or .tiff/.png (Export).
        """
        if current_depth_mm is None:
            return False
            
        try:
            valid_depth = np.nan_to_num(current_depth_mm, nan=0.0, posinf=0.0, neginf=0.0)

            # --- OPTION A: .npy (Native Python) ---
            # Best for saving/loading within your own app
            if file_path.endswith('.npy'):
                np.save(file_path, valid_depth.astype(np.float32))
                print(f"[DEM] Saved binary surface data to {file_path}")
                return True

            # --- OPTION B: TIFF/PNG (Export) ---
            # Create a visible preview first (optional but helpful)
            preview_path = file_path.rsplit('.', 1)[0] + "_preview.png"
            d_min, d_max = np.min(valid_depth), np.max(valid_depth)
            norm_range = d_max - d_min
            if norm_range > 0:
                preview = ((valid_depth - d_min) / norm_range * 255.0).astype(np.uint8)
                cv2.imwrite(preview_path, cv2.applyColorMap(preview, cv2.COLORMAP_JET))

            # Try to save the main file
            if file_path.endswith('.tiff') or file_path.endswith('.tif'):
                # Try saving as float32. If this crashes later on load, use .npy!
                return cv2.imwrite(file_path, valid_depth.astype(np.float32))
            else:
                # Default to 16-bit PNG (Robust and standard)
                return cv2.imwrite(file_path, valid_depth.astype(np.uint16))

        except Exception as e:
            print(f"[DEM] Save Error: {e}")
            return False

    def toggle(self):
        if self.dem_loaded:
            self.active = not self.active
            return self.active
        return False
    

    def compute_guidance_layer(self, current_depth_mm):
        """
        Returns a BGR overlay image showing the difference between sand and DEM.
        Red = Sand is too high (Remove sand)
        Blue = Sand is too low (Add sand)
        Green = Match
        """
        if not self.active or self.dem_grid is None:
            return None

        # Calculate difference: (Current Sand) - (Target DEM)
        # Note: Depending on your camera setup, 'higher' Z value might mean 'deeper'.
        # Assuming: Higher Z value = Deeper (further from camera)
        # Therefore: If Sand(800mm) - DEM(900mm) = -100mm. Sand is physically HIGHER (closer to cam).
        
        diff = current_depth_mm - self.dem_grid

        # Initialize output image
        output = np.zeros((self.target_h, self.target_w, 3), dtype=np.uint8)

        # Create Masks
        # 1. Match (within tolerance) -> Green
        mask_match = np.abs(diff) < self.tolerance
        
        # 2. Sand is higher (closer to camera, smaller depth value) than target -> Red (Dig)
        # If camera measures 800mm and target is 1000mm, diff is -200. 
        mask_high = diff < -self.tolerance
        
        # 3. Sand is lower (deeper, larger depth value) than target -> Blue (Fill)
        mask_low = diff > self.tolerance

        # Apply Colors (BGR)
        output[mask_match] = [0, 255, 0]   # Green
        output[mask_high]  = [0, 0, 255]   # Red
        output[mask_low]   = [255, 0, 0]   # Blue
        
        # Optional: Gradient intensity based on how far off it is
        # normalize errors for gradient effect (clamped at 100mm error)
        # error_magnitude = np.clip(np.abs(diff) / 100.0, 0, 1)
        # output = (output * error_magnitude[:,:,np.newaxis]).astype(np.uint8)

        return output
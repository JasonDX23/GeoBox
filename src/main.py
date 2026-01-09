import sys
import threading
import time
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject

# --- Internal Module Imports ---
from core.kinect import KinectDevice
from core.calibration import ProjectorCalibrator
from core.processor import DepthProcessor
from core.config import ConfigManager
from modules.color_maps import HybridColorMapper
from modules.fluid_sim import FluidSimulator
from ui.window import RenderWindow
from ui.dashboard import GeoBoxDashboard

class EngineSignals(QObject):
    """Signals sent from the Engine thread to the GUI thread."""
    frame_ready = Signal(np.ndarray)

class GeoBoxEngine(threading.Thread):
    def __init__(self, signals, dash_signals):
        super().__init__()
        self.daemon = True
        self.running = False
        self.signals = signals
        
        # 1. Load Persistence Layer
        self.config_mgr = ConfigManager()
        self.conf_data = self.config_mgr.load()
        
        # 2. Initialize Subsystems
        self.calibrator = ProjectorCalibrator()
        # Load saved ROI and 2.5D Matrix
        self.calibrator.set_config(self.conf_data["roi"], self.conf_data["matrix"])
        
        self.processor = DepthProcessor(
            min_depth=self.conf_data["depth_range"][0], 
            max_depth=self.conf_data["depth_range"][1]
        )
        self.mapper = HybridColorMapper()
        self.fluids = FluidSimulator()
        
        # 3. State Machine Variables
        self.mode = "RUN"          # Modes: RUN, ROI, CALIB_SEARCH, CALIB_WAIT
        self.calib_stage = "BASE"  # Stages: BASE, TOP
        
        self.sea_offset = 0.0
        self.contours_on = False
        self.contour_int = 0.05
        
        # 4. Connect GUI Signals to Engine Slots
        # --- Visualization ---
        dash_signals.sea_changed.connect(self.set_sea)
        dash_signals.mode_preset.connect(self.set_preset)
        dash_signals.mode_custom.connect(self.set_custom)
        dash_signals.contours_toggled.connect(self.toggle_c)
        dash_signals.contour_interval.connect(self.set_c_int)
        
        # --- Fluids ---
        dash_signals.rain_changed.connect(self.set_rain)
        dash_signals.evap_changed.connect(self.set_evap)
        dash_signals.clear_water.connect(self.clear_water)
        
        # --- Calibration ---
        dash_signals.start_roi_select.connect(self.set_roi_mode)
        dash_signals.capture_base.connect(self.start_capture_base)
        dash_signals.capture_top.connect(self.start_capture_top)
        dash_signals.save_calibration.connect(self.compute_and_save)

    # ========================== SLOTS ==========================
    
    # Visualization Setters
    def set_sea(self, v): self.sea_offset = v
    def set_preset(self, n): self.mapper.set_mode_preset(n)
    def set_custom(self, s): self.mapper.set_mode_custom(s)
    def toggle_c(self, b): self.contours_on = b
    def set_c_int(self, v): self.contour_int = v
    
    # Fluid Setters
    def set_rain(self, on, rate): self.fluids.set_rain(on, rate)
    def set_evap(self, rate): self.fluids.evaporation_rate = rate
    def clear_water(self): self.fluids.water_depth[:] = 0

    # Calibration Logic
    def set_roi_mode(self):
        print("[Engine] Mode: ROI Selection")
        self.mode = "ROI"
    
    def update_roi(self, roi_list):
        print(f"[Engine] ROI Updated: {roi_list}")
        self.calibrator.roi = roi_list
        self.mode = "RUN" # Return to simulation after selection

    def start_capture_base(self):
        print("[Engine] Calibration: Capturing BASE Level...")
        self.calibrator.reset_collection() # Clear old data
        self.calib_stage = "BASE"
        self.mode = "CALIB_SEARCH"

    def start_capture_top(self):
        print("[Engine] Calibration: Capturing TOP Level...")
        # Do not reset collection (we need both sets of points)
        self.calib_stage = "TOP"
        self.mode = "CALIB_SEARCH"

    def compute_and_save(self):
        print("[Engine] Computing 2.5D Regression...")
        success = self.calibrator.compute_calibration()
        if success:
            print("[Engine] Calibration Successful. Saving...")
            self.config_mgr.save(
                self.calibrator.roi, 
                self.calibrator.matrix,
                [self.processor.min_d, self.processor.max_d]
            )
        else:
            print("[Engine] Calibration Failed (Not enough points).")
        self.mode = "RUN"

    # ========================== MAIN LOOP ==========================

    def run(self):
        print("[Engine] GeoBox Kernel Started.")
        
        # Hardware Initialization
        try:
            kinect = KinectDevice()
        except Exception as e:
            print(f"[Engine] HARDWARE FATAL: {e}")
            return

        proj_win = RenderWindow("GeoBox Projector")
        self.running = True
        
        while self.running:
            try:
                # --- 1. ACQUISITION ---
                # Retrieve the correct frame type based on mode
                if self.mode in ["CALIB_SEARCH", "CALIB_WAIT"]:
                    # RGB needed for Checkerboard
                    rgb_frame = kinect.get_rgb_frame()
                    depth_frame = kinect.get_depth_frame() # Depth also needed for Z-correlation
                else:
                    # Simulation only needs Depth
                    depth_frame = kinect.get_depth_frame()

                # --- 2. PROCESSING PIPELINE ---
                
                if self.mode == "CALIB_SEARCH":
                    # A. Generate Pattern
                    pattern = self.calibrator.generate_pattern()
                    
                    # B. Add Instruction Text to Projection
                    msg = f"CAPTURING: {self.calib_stage}"
                    cv2.putText(pattern, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
                    proj_win.show(pattern)
                    
                    # C. Detect Corners
                    success, debug_viz = self.calibrator.capture_points(rgb_frame, depth_frame)
                    
                    # Send feedback to Dashboard
                    self.signals.frame_ready.emit(debug_viz)
                    
                    if success:
                        print(f"[Engine] {self.calib_stage} Points Acquired.")
                        self.mode = "CALIB_WAIT" # Pause capturing to avoid duplicate data

                elif self.mode == "CALIB_WAIT":
                    # Just show the pattern, don't process. 
                    # Waiting for user to click next button.
                    pattern = self.calibrator.generate_pattern()
                    cv2.putText(pattern, f"{self.calib_stage} CAPTURED - NEXT STEP?", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 3)
                    proj_win.show(pattern)
                    
                    # Send the last captured RGB frame just to keep UI alive
                    self.signals.frame_ready.emit(rgb_frame)

                elif self.mode == "ROI":
                    # Show User the raw depth to draw the box
                    # Normalize 11-bit (0-2047) to 8-bit visual
                    viz = (depth_frame / 2048.0 * 255).astype(np.uint8)
                    viz_color = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
                    
                    # Project Black (to not interfere with visibility)
                    black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
                    proj_win.show(black_screen)
                    
                    # Send Feed to UI
                    self.signals.frame_ready.emit(viz_color)

                elif self.mode == "RUN":
                    # --- A. Calibration Warp (2.5D) ---
                    warped_depth = self.calibrator.warp(depth_frame)
                    
                    # --- B. Normalization & In-Painting ---
                    terrain_height = self.processor.normalize(warped_depth)
                    
                    # --- C. Physics Simulation ---
                    fluid_map = self.fluids.step(terrain_height)
                    
                    # --- D. Rendering ---
                    # 1. Terrain Gradient
                    frame = self.mapper.apply(terrain_height, self.sea_offset)
                    
                    # 2. Fluid Overlay
                    if np.any(fluid_map > 0.01):
                        # Create water color layer
                        water_color = np.full_like(frame, [200, 100, 0]) # Deep Blue
                        
                        # Calculate Alpha (Deeper = More Opaque)
                        alpha = np.clip(fluid_map * 4.0, 0, 0.75)
                        alpha = np.dstack([alpha, alpha, alpha])
                        
                        # Alpha Blend
                        f_frame = frame.astype(np.float32)
                        f_water = water_color.astype(np.float32)
                        blended = f_frame * (1.0 - alpha) + f_water * alpha
                        frame = blended.astype(np.uint8)
                    
                    # 3. Contour Overlay
                    if self.contours_on:
                        c_layer = self.processor.get_contour_layer(terrain_height, self.contour_int)
                        if c_layer is not None:
                            # c_layer is BGRA. Extract Alpha.
                            alpha_c = c_layer[:, :, 3] / 255.0
                            inv_alpha = 1.0 - alpha_c
                            
                            for c in range(3):
                                frame[:, :, c] = (alpha_c * c_layer[:, :, c] + 
                                                  inv_alpha * frame[:, :, c]).astype(np.uint8)

                    # --- E. Output ---
                    proj_win.show(frame)
                    self.signals.frame_ready.emit(frame)

            except Exception as e:
                # Catch transient errors (e.g., USB frame drop) to prevent crash
                print(f"[Engine] Loop Warning: {e}")
                continue

            # Handle Window Close Events
            if not proj_win.process_input():
                self.running = False

        # Shutdown Sequence
        kinect.close()
        proj_win.destroy()
        print("[Engine] Shutdown Complete.")

def main():
    # 1. Setup Qt Application
    app = QApplication(sys.argv)
    
    # 2. Initialize Dashboard
    dashboard = GeoBoxDashboard()
    
    # 3. Initialize Engine with Signal Bridge
    eng_signals = EngineSignals()
    engine = GeoBoxEngine(eng_signals, dashboard.signals)
    
    # 4. Critical Wiring: ROI Overlay -> Engine
    # This connects the transparent drawing widget directly to the engine
    dashboard.roi_overlay.roi_selected.connect(engine.update_roi)
    
    # 5. Engine -> Dashboard (Video Feed)
    eng_signals.frame_ready.connect(dashboard.update_feed)
    
    # 6. Launch
    dashboard.show()
    engine.start()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
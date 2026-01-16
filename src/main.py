import sys
import time
import traceback
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QThread, Signal, QObject

# --- Custom Modules ---
from core.kinect import KinectDevice
from core.calibration import ProjectorCalibrator
from core.processor import DepthProcessor
from core.config import ConfigManager
from modules.color_maps import HybridColorMapper
from modules.fluid_sim import FluidSimulator
from ui.window import RenderWindow
from ui.dashboard import GeoBoxDashboard

class GeoBoxEngine(QThread):
    """
    The Core Physics & Rendering Engine.
    Runs on a separate thread to keep the GUI responsive.
    """
    frame_ready = Signal(np.ndarray)      # Sends final image to Dashboard
    range_auto_set = Signal(int, int)     # Updates UI sliders after auto-level

    def __init__(self, dash_signals):
        super().__init__()
        self.setTerminationEnabled(True)

        # 1. Load Configuration
        self.config_mgr = ConfigManager()
        self.conf_data = self.config_mgr.load()

        # 2. Initialize Logic Modules
        self.calibrator = ProjectorCalibrator()
        self.calibrator.set_config(self.conf_data["roi"], self.conf_data["matrix"])
        
        self.processor = DepthProcessor(
            min_depth=self.conf_data["depth_range"][0], 
            max_depth=self.conf_data["depth_range"][1]
        )
        self.mapper = HybridColorMapper()
        self.mapper.load_cpt_file("HeightColorMap.cpt")
        self.fluids = FluidSimulator()

        # 3. Internal State
        self.running = False
        self.mode = "RUN"
        self.wizard_step = 0
        self.calib_trigger = False
        self.flash_timer = 0
        self.flash_type = 1 
        self.perform_auto_level = False
        
        # Visual/Simulation Parameters
        self.sea_offset = 0.0
        self.contours_on = False
        self.contour_int = 0.05
        self.flip_orientation = False 
        self.output_scale_x = 1.0
        self.output_scale_y = 1.0
        self.calculate_scaling() 

        # 4. Wire Dashboard Inputs -> Engine Slots
        dash_signals.sea_changed.connect(self.set_sea)
        dash_signals.flip_changed.connect(self.set_flip) 
        dash_signals.mode_preset.connect(self.set_preset)
        dash_signals.mode_custom.connect(self.set_custom)
        dash_signals.contours_toggled.connect(self.toggle_c)
        dash_signals.contour_interval.connect(self.set_c_int)
        dash_signals.depth_range_update.connect(self.update_depth_range)
        dash_signals.trigger_auto_level.connect(self.trigger_auto) 
        dash_signals.rain_changed.connect(self.set_rain)
        dash_signals.evap_changed.connect(self.set_evap)
        dash_signals.clear_water.connect(self.clear_water)
        dash_signals.start_roi_select.connect(self.set_roi_mode)
        dash_signals.calib_start.connect(self.wiz_start)
        dash_signals.calib_next.connect(self.wiz_next)
        dash_signals.calib_finish.connect(self.wiz_finish)

    # --- SETTERS / SLOTS ---
    def set_sea(self, v): self.sea_offset = v
    def set_flip(self, b): self.flip_orientation = b
    def set_preset(self, n): self.mapper.set_mode_preset(n)
    def set_custom(self, s): self.mapper.set_mode_custom(s)
    def toggle_c(self, b): self.contours_on = b
    def set_c_int(self, v): self.contour_int = v
    def update_depth_range(self, mn, mx): self.processor.min_d, self.processor.max_d = mn, mx
    def set_rain(self, on, rate): self.fluids.set_rain(on, rate)
    def set_evap(self, rate): self.fluids.evaporation_rate = rate
    def clear_water(self): self.fluids.water_depth[:] = 0
    def set_roi_mode(self): self.mode = "ROI"
    def wiz_start(self): self.calibrator.reset_collection(); self.wizard_step = 0; self.mode = "CALIB_WIZARD"
    def wiz_next(self): self.calib_trigger = True
    def trigger_auto(self): self.perform_auto_level = True

    def update_roi(self, roi): 
        self.calibrator.roi = roi
        self.calculate_scaling() 
        self.config_mgr.save(self.calibrator.roi, self.calibrator.matrix, [self.processor.min_d, self.processor.max_d])
        self.mode = "RUN"

    def wiz_finish(self): 
        if self.calibrator.compute(): 
            self.config_mgr.save(self.calibrator.roi, self.calibrator.matrix, [self.processor.min_d, self.processor.max_d])
        self.mode = "RUN"

    def calculate_scaling(self):
        rx, ry, rw, rh = self.calibrator.roi
        if rw < 10 or rh < 10: 
            self.output_scale_x = 1.0; self.output_scale_y = 1.0; return
        self.output_scale_x = 640.0 / float(rw)
        self.output_scale_y = 480.0 / float(rh)

    # --- MOUSE INTERACTION ---
    def mouse_callback(self, event, x, y, flags, param):
        """
        Catches mouse clicks on the projector window 
        and updates the fluid simulator's cursor.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.fluids.set_interaction_point(x, y, True)
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
            self.fluids.set_interaction_point(x, y, True)
        elif event == cv2.EVENT_LBUTTONUP:
            self.fluids.set_interaction_point(x, y, False)

    # --- WORKER LOOP ---
    def run(self):
        print("[Engine] GeoBox Kernel Starting...")
        
        kinect = None
        proj_win = None
        
        try:
            import freenect 
            kinect = KinectDevice()
            print("[Engine] Kinect Hardware Connected.")
        except Exception as e:
            print(f"[CRITICAL] Kinect Init Failed: {e}")
            traceback.print_exc()
            return 

        try:
            proj_win = RenderWindow("GeoBox Projector")
            
            # --- NEW: Enable Mouse Interaction ---
            # Create window immediately to attach callback
            cv2.namedWindow("GeoBox Projector", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("GeoBox Projector", self.mouse_callback)
            
        except Exception as e:
            print(f"[CRITICAL] Window Init Failed: {e}")
            if kinect: kinect.close()
            return

        self.running = True

        while self.running:
            try:
                # --- A. DATA ACQUISITION ---
                d_raw = kinect.get_depth_frame()
                if d_raw is None:
                    self.msleep(5) 
                    continue

                dashboard_viz = None 

                # --- B. MODE SWITCHING ---
                
                # [MODE 1] CALIBRATION WIZARD
                if self.mode == "CALIB_WIZARD":
                    rgb = kinect.get_rgb_frame()
                    if rgb is not None:
                        pat = self.calibrator.generate_dynamic_pattern(self.wizard_step)
                        
                        if self.flash_timer > 0:
                            overlay = np.zeros_like(pat)
                            color = (0, 255, 0) if self.flash_type == 1 else (0, 0, 255)
                            overlay[:] = color
                            pat = cv2.addWeighted(pat, 0.7, overlay, 0.3, 0)
                            self.flash_timer -= 1
                        
                        proj_win.show(pat)
                        
                        if self.calib_trigger:
                            success, viz = self.calibrator.capture_frame(rgb, d_raw, self.wizard_step)
                            dashboard_viz = viz
                            if success:
                                self.wizard_step += 1
                                self.calib_trigger = False
                                self.flash_timer = 10
                                self.flash_type = 1 
                                if self.wizard_step >= 10: self.mode = "CALIB_WAIT"
                            else:
                                self.calib_trigger = False
                                self.flash_timer = 10
                                self.flash_type = -1 
                        else:
                            dashboard_viz = rgb

                # [MODE 2] WAIT FOR COMPUTE
                elif self.mode == "CALIB_WAIT":
                    img = np.zeros((480,640,3), dtype=np.uint8)
                    cv2.putText(img, "DONE - CLICK COMPUTE", (50,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    proj_win.show(img)
                    rgb = kinect.get_rgb_frame()
                    if rgb is not None: dashboard_viz = rgb

                # [MODE 3] ROI SELECTION
                elif self.mode == "ROI":
                    viz = (d_raw / 2048.0 * 255).astype(np.uint8)
                    dashboard_viz = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
                    proj_win.show(np.zeros((480, 640, 3), dtype=np.uint8))

                # [MODE 4] RUN SIMULATION (Main Loop)
                elif self.mode == "RUN":
                    # 1. Hard ROI Crop
                    rx, ry, rw, rh = self.calibrator.roi
                    masked_raw = np.ones_like(d_raw) * 2047 
                    rx, ry = max(0, rx), max(0, ry)
                    rw, rh = min(rw, 640-rx), min(rh, 480-ry)
                    
                    if rw > 0 and rh > 0:
                        masked_raw[ry:ry+rh, rx:rx+rw] = d_raw[ry:ry+rh, rx:rx+rw]

                    # 2. Calibration Warp
                    warped = self.calibrator.warp(masked_raw)
                    if self.flip_orientation: warped = cv2.flip(warped, -1)
                    
                    # 3. Post-Scale 
                    warped_cropped = warped[ry:ry+rh, rx:rx+rw]
                    if warped_cropped.size == 0: continue
                    
                    warped_scaled = cv2.resize(warped_cropped, (640, 480), interpolation=cv2.INTER_LINEAR)
                    
                    # 4. Auto Level
                    if self.perform_auto_level:
                        mn, mx = self.processor.auto_range(warped_scaled)
                        self.processor.min_d, self.processor.max_d = mn, mx
                        self.range_auto_set.emit(mn, mx)
                        self.perform_auto_level = False
                    
                    # 6. Height Map Generation
                    vis_height = self.processor.normalize(warped_scaled)
                    phys_height = self.processor.normalize_for_physics(warped_scaled)
                    
                    # 7. Physics Step (Returns Electric Blue Image)
                    fluid_visual = self.fluids.step(phys_height)
                    
                    # 8. Rendering
                    frame = self.mapper.apply(vis_height, self.sea_offset)
                    
                    # 9. Overlay Fluid
                    if np.any(fluid_visual > 0):
                        # Create an alpha mask based on water brightness
                        water_gray = cv2.cvtColor(fluid_visual, cv2.COLOR_BGR2GRAY)
                        # Cap opacity at 80% to ensure sand is visible under water
                        alpha = np.clip(water_gray / 100.0, 0, 0.8)
                        alpha = alpha[:, :, np.newaxis] 
                        
                        # Blend: Frame * (1-alpha) + Water * alpha
                        frame = (frame * (1.0 - alpha) + fluid_visual * alpha).astype(np.uint8)
                    
                    # Overlay Contours
                    if self.contours_on:
                        cl = self.processor.get_contour_layer(vis_height, self.contour_int)
                        if cl is not None:
                            ac = cl[:,:,3]/255.0
                            for c in range(3):
                                frame[:,:,c] = (ac*cl[:,:,c] + (1-ac)*frame[:,:,c]).astype(np.uint8)

                    # HUD Overlay
                    try:
                        h_in, w_in = warped_scaled.shape
                        val = warped_scaled[h_in//2, w_in//2]
                        depth_mm = int(val) if np.isfinite(val) else 0
                    except: pass

                    proj_win.show(frame)
                    dashboard_viz = frame

                # --- C. UPDATE UI ---
                if dashboard_viz is not None:
                    self.frame_ready.emit(dashboard_viz)

                # --- D. WINDOW EVENTS ---
                if not proj_win.process_input():
                    self.running = False

            except Exception as e:
                print(f"[Loop Error] {e}")
                traceback.print_exc()
                self.msleep(100) 

        if kinect: kinect.close()
        if proj_win: proj_win.destroy()
        print("[Engine] GeoBox Kernel Stopped.")

def main():
    app = QApplication(sys.argv)
    
    dash = GeoBoxDashboard()
    engine = GeoBoxEngine(dash.signals)
    
    engine.frame_ready.connect(dash.update_feed)
    engine.range_auto_set.connect(dash.set_depth_sliders)

    if hasattr(dash, 'video'):
         dash.video.roi_selected.connect(engine.update_roi)
    else:
         print("[WARNING] dash.video not found. ROI selection may not work.")
    
    dash.show()
    engine.start() 
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
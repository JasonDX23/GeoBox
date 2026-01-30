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
from modules.dem_loader import DEMHandler
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
        # Ensure calibrator is initialized with projector resolution
        self.calibrator = ProjectorCalibrator(width=1024, height=768)
        self.calibrator.set_config(self.conf_data["roi"], self.conf_data["matrix"])
        
        self.processor = DepthProcessor(
            min_depth=self.conf_data["depth_range"][0], 
            max_depth=self.conf_data["depth_range"][1]
        )
        self.mapper = HybridColorMapper()
        self.mapper.load_cpt_file("HeightColorMap.cpt")
        self.fluids = FluidSimulator()
        
        self.dem_handler = DEMHandler(target_width=640, target_height=480)
        self.current_phys_height = None

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
        
        dash_signals.dem_load_request.connect(self.load_dem_request)
        dash_signals.dem_save_request.connect(self.save_dem_request)
        dash_signals.dem_toggle.connect(self.toggle_dem)

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

    def load_dem_request(self, filepath):
        self.dem_handler.load_dem(filepath, self.processor.min_d, self.processor.max_d)

    def save_dem_request(self, filepath):
        if self.current_phys_height is not None:
             self.dem_handler.save_dem(filepath, self.current_phys_height)

    def toggle_dem(self, active):
        self.dem_handler.active = active

    def update_roi(self, roi): 
        self.calibrator.roi = roi
        self.config_mgr.save(self.calibrator.roi, self.calibrator.matrix, [self.processor.min_d, self.processor.max_d])
        self.mode = "RUN"

    def wiz_finish(self): 
        if self.calibrator.compute(): 
            self.config_mgr.save(self.calibrator.roi, self.calibrator.matrix, [self.processor.min_d, self.processor.max_d])
        self.mode = "RUN"

    def mouse_callback(self, event, x, y, flags, param):
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
            from core.kinect import KinectDevice
            kinect = KinectDevice()
            print("[Engine] Kinect Hardware Connected.")
            proj_win = RenderWindow("GeoBox Projector")
            cv2.namedWindow("GeoBox Projector", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback("GeoBox Projector", self.mouse_callback)
        except Exception as e:
            print(f"[CRITICAL] Initialization Failed: {e}")
            traceback.print_exc()
            return

        self.running = True

        while self.running:
            try:
                d_raw = kinect.get_depth_frame()
                if d_raw is None:
                    self.msleep(5); continue

                dashboard_viz = None 

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
                    proj_win.show(np.zeros((768, 1024, 3), dtype=np.uint8))

                # [MODE 4] RUN SIMULATION
                elif self.mode == "RUN":
                    # --- SCALING FIX ---
                    # The 11-coeff warp returns a 1024x768 frame automatically.
                    # We no longer manually crop or resize back to 640x480.
                    warped = self.calibrator.warp(d_raw)
                    
                    if self.flip_orientation: 
                        warped = cv2.flip(warped, -1)
                    
                    if self.perform_auto_level:
                        mn, mx = self.processor.auto_range(warped)
                        self.processor.min_d, self.processor.max_d = mn, mx
                        self.range_auto_set.emit(mn, mx)
                        self.perform_auto_level = False
                    
                    vis_height = self.processor.normalize(warped)
                    phys_height = self.processor.normalize_for_physics(warped)
                    self.current_phys_height = phys_height
                    
                    fluid_visual = self.fluids.step(phys_height)
                    frame = self.mapper.apply(vis_height, self.sea_offset)
                    
                    if self.dem_handler.active:
                        gl = self.dem_handler.compute_guidance_layer(phys_height)
                        if gl is not None:
                            frame = cv2.addWeighted(frame, 0.6, gl, 0.4, 0)
                    
                    if np.any(fluid_visual > 0):
                        water_gray = cv2.cvtColor(fluid_visual, cv2.COLOR_BGR2GRAY)
                        alpha = np.clip(water_gray / 100.0, 0, 0.8)[:, :, np.newaxis] 
                        frame = (frame * (1.0 - alpha) + fluid_visual * alpha).astype(np.uint8)
                    
                    if self.contours_on:
                        cl = self.processor.get_contour_layer(vis_height, self.contour_int)
                        if cl is not None:
                            ac = cl[:,:,3]/255.0
                            for c in range(3):
                                frame[:,:,c] = (ac*cl[:,:,c] + (1-ac)*frame[:,:,c]).astype(np.uint8)

                    proj_win.show(frame)
                    dashboard_viz = frame

                if dashboard_viz is not None:
                    self.frame_ready.emit(dashboard_viz)

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
    dash.show()
    engine.start() 
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
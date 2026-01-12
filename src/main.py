import sys
import threading
import time
import numpy as np
import cv2
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, QObject

# Modules
from core.kinect import KinectDevice
from core.calibration import ProjectorCalibrator
from core.processor import DepthProcessor
from core.config import ConfigManager
from modules.color_maps import HybridColorMapper
from modules.fluid_sim import FluidSimulator
from ui.window import RenderWindow
from ui.dashboard import GeoBoxDashboard

class EngineSignals(QObject):
    frame_ready = Signal(np.ndarray)
    range_auto_set = Signal(int, int)

class GeoBoxEngine(threading.Thread):
    def __init__(self, signals, dash_signals):
        super().__init__()
        self.daemon = True
        self.running = False
        self.signals = signals
        
        # Load Config
        self.config_mgr = ConfigManager()
        self.conf_data = self.config_mgr.load()
        
        # Initialize Core Modules
        self.calibrator = ProjectorCalibrator()
        self.calibrator.set_config(self.conf_data["roi"], self.conf_data["matrix"])
        
        self.processor = DepthProcessor(
            min_depth=self.conf_data["depth_range"][0], 
            max_depth=self.conf_data["depth_range"][1]
        )
        self.mapper = HybridColorMapper()
        self.fluids = FluidSimulator()
        
        # State Flags
        self.mode = "RUN"
        self.wizard_step = 0
        self.calib_trigger = False
        self.flash_timer = 0
        self.flash_type = 1 # 1=Green, -1=Red
        self.perform_auto_level = False
        
        # Visual Settings
        self.sea_offset = 0.0
        self.contours_on = False
        self.contour_int = 0.05
        self.flip_orientation = False 
        
        # Scaling State
        self.output_scale_x = 1.0
        self.output_scale_y = 1.0
        self.calculate_scaling() 
        
        # Signal Connections
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

    # --- Slots ---
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

    # --- Main Loop ---
    def run(self):
        print("[Engine] GeoBox Kernel Started.")
        kinect = None
        try: kinect = KinectDevice()
        except: return

        proj_win = RenderWindow("GeoBox Projector")
        self.running = True
        
        while self.running:
            try:
                # 1. ACQUIRE
                d_raw = kinect.get_depth_frame()
                if d_raw is None: 
                    time.sleep(0.01)
                    continue

                # --- MODE: CALIBRATION WIZARD ---
                if self.mode == "CALIB_WIZARD":
                    rgb = kinect.get_rgb_frame()
                    if rgb is None: continue
                    
                    pat = self.calibrator.generate_dynamic_pattern(self.wizard_step)
                    phase = "SAND" if self.wizard_step < 5 else "BOARD"
                    # msg = f"STEP {self.wizard_step+1}/10 - {phase}"
                    # cv2.putText(pat, msg, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    
                    # Flash Feedback
                    if self.flash_timer > 0:
                        overlay = np.zeros_like(pat)
                        color = (0, 255, 0) if self.flash_type == 1 else (0, 0, 255)
                        overlay[:,:] = color
                        pat = cv2.addWeighted(pat, 0.7, overlay, 0.3, 0)
                        self.flash_timer -= 1
                    
                    proj_win.show(pat)
                    
                    if self.calib_trigger:
                        # Pass step_idx so calibrator knows if we are in "Blind Board" phase
                        success, viz = self.calibrator.capture_frame(rgb, d_raw, self.wizard_step)
                        self.signals.frame_ready.emit(viz)
                        if success:
                            self.wizard_step += 1
                            self.calib_trigger = False
                            self.flash_timer = 10
                            self.flash_type = 1 # Green
                            if self.wizard_step >= 10: self.mode = "CALIB_WAIT"
                        else:
                             self.calib_trigger = False
                             self.flash_timer = 10
                             self.flash_type = -1 # Red
                    else:
                        self.signals.frame_ready.emit(rgb)

                # --- MODE: WAIT FOR COMPUTE ---
                elif self.mode == "CALIB_WAIT":
                    img = np.zeros((480,640,3), dtype=np.uint8)
                    cv2.putText(img, "DONE - CLICK COMPUTE", (50,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                    proj_win.show(img)
                    if kinect.get_rgb_frame() is not None:
                        self.signals.frame_ready.emit(kinect.get_rgb_frame())

                # --- MODE: ROI SELECTION ---
                elif self.mode == "ROI":
                    viz = (d_raw / 2048.0 * 255).astype(np.uint8)
                    viz_color = cv2.applyColorMap(viz, cv2.COLORMAP_JET)
                    proj_win.show(np.zeros((480, 640, 3), dtype=np.uint8))
                    self.signals.frame_ready.emit(viz_color)

                # --- MODE: RUN SIMULATION ---
                elif self.mode == "RUN":
                    # 1. Hard ROI Crop (Delete outside noise)
                    rx, ry, rw, rh = self.calibrator.roi
                    masked_raw = np.ones_like(d_raw) * 2047 
                    rx, ry = max(0, rx), max(0, ry)
                    rw, rh = min(rw, 640-rx), min(rh, 480-ry)
                    if rw > 0 and rh > 0:
                        masked_raw[ry:ry+rh, rx:rx+rw] = d_raw[ry:ry+rh, rx:rx+rw]

                    # 2. Calibration Warp (Rectify Image)
                    warped = self.calibrator.warp(masked_raw)
                    if self.flip_orientation: warped = cv2.flip(warped, -1)
                    
                    # 3. Post-Scale (Zoom to fill screen)
                    warped_cropped = warped[ry:ry+rh, rx:rx+rw]
                    if warped_cropped.size == 0: continue
                    warped_scaled = cv2.resize(warped_cropped, (640, 480), interpolation=cv2.INTER_LINEAR)
                    
                    # 4. Auto Level (Adjust Colors)
                    if self.perform_auto_level:
                        mn, mx = self.processor.auto_range(warped_scaled)
                        self.processor.min_d, self.processor.max_d = mn, mx
                        self.signals.range_auto_set.emit(mn, mx)
                        self.perform_auto_level = False
                    
                    # 5. Hand Rain Interaction
                    # We use warped_scaled to detect hand positions relative to sand
                    self.fluids.apply_hands(warped_scaled, sand_min_threshold=self.processor.min_d - 50)

                    # 6. Generate Height Maps
                    # A. Visuals: Includes hands (for rendering)
                    vis_height = self.processor.normalize(warped_scaled)
                    
                    # B. Physics: Hands Removed (so water falls through)
                    phys_height = self.processor.normalize_for_physics(warped_scaled)
                    
                    # 7. Simulation Step (Uses Physics Map)
                    fluid = self.fluids.step(phys_height)
                    
                    # 8. Rendering (Uses Visual Map)
                    frame = self.mapper.apply(vis_height, self.sea_offset)
                    
                    # Overlay Fluid
                    if np.any(fluid > 0.01):
                        wc = np.full_like(frame, [200, 100, 0])
                        alpha = np.dstack([np.clip(fluid*4,0,0.7)]*3)
                        frame = (frame*(1-alpha) + wc*alpha).astype(np.uint8)
                    
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
                        hud_text = f"REAL: {depth_mm}mm | MIN: {self.processor.min_d} | MAX: {self.processor.max_d}"
                        cv2.putText(frame, hud_text, (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    except: pass

                    proj_win.show(frame)
                    self.signals.frame_ready.emit(frame)

            except Exception as e:
                # print(e) # Suppress loop errors for stability
                continue
            
            if not proj_win.process_input(): self.running = False

        if kinect: kinect.close()
        proj_win.destroy()

def main():
    app = QApplication(sys.argv)
    dash = GeoBoxDashboard()
    eng_signals = EngineSignals()
    engine = GeoBoxEngine(eng_signals, dash.signals)
    
    # Connect Dashboard <-> Engine
    dash.video.roi_selected.connect(engine.update_roi)
    eng_signals.frame_ready.connect(dash.update_feed)
    eng_signals.range_auto_set.connect(dash.set_depth_sliders)
    
    dash.show()
    engine.start()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
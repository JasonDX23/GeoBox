Here is the comprehensive technical documentation for **GeoBox**. This document is designed to serve as the "Architectural Bible" for the repository, enabling any developer to understand, maintain, and extend the system.

---

# GeoBox: Augmented Reality Sandbox Engine

**Technical Documentation & Developer Guide**

## 1. System Overview

**GeoBox** is a Python-based engine for driving an Augmented Reality (AR) Sandbox. It bridges the gap between physical topography (sand) and digital simulation (water/terrain projection).

Unlike game-engine based solutions (Unity/Unreal), GeoBox is built "from scratch" using **NumPy**, **OpenCV**, and **PySide6 (Qt)**. This ensures low overhead, direct access to depth buffers, and a customizable Pythonic codebase.

### Core Philosophy

1. **Raw Data Access:** We interact directly with the Kinect depth map (`uint16` array).
2. **Vectorized Physics:** Fluid simulations utilize NumPy array operations rather than object-based agents for performance.
3. **Thread Safety:** The heavy computer vision loop runs on a dedicated worker thread, keeping the GUI responsive.
4. **2.5D Calibration:** We solve for parallax correction mathematically, removing the need for 3D meshes or vertex shaders.

---

## 2. Architecture & Data Flow

The system follows a **Producer-Consumer** pattern wrapped in a threaded architecture.

### High-Level Data Pipeline

1. **Hardware Interface (`KinectDevice`)**: Fetches raw depth (11-bit) and RGB frames.
2. **Calibration Layer (`ProjectorCalibrator`)**: Corrects geometric distortion and parallax shift using a 2.5D Regression Matrix.
3. **Processing Layer (`DepthProcessor`)**:
* **Temporal Smoothing:** Reduces sensor jitter.
* **In-Painting:** Fills "shadow" holes caused by the Kinect's IR offset.
* **Normalization:** Converts physical depth (mm) to a 0.0–1.0 height map.


4. **Simulation Layer (`FluidSimulator`)**: Calculates water flow based on the height map.
5. **Rendering Layer**: Applies color gradients (Matplotlib/Custom), blends fluid layers, and draws vector contours.
6. **Output**:
* **Projector Window:** Fullscreen display on the secondary monitor.
* **Dashboard Preview:** Downscaled feedback sent to the GUI via Qt Signals.



---

## 3. Directory Structure

```text
GeoBox/
├── src/
│   ├── main.py                 # Entry Point & State Machine (The Brain)
│   ├── core/                   # Low-level Logic
│   │   ├── kinect.py           # libfreenect wrapper
│   │   ├── calibration.py      # 2.5D Regression & checkerboard logic
│   │   ├── processor.py        # Depth filtering & Contour gen
│   │   └── config.py           # JSON Persistence Manager
│   ├── modules/                # High-level Features
│   │   ├── fluid_sim.py        # Virtual Pipes (Hydrostatic) Physics
│   │   └── color_maps.py       # Gradient Generation (Hybrid Mode)
│   └── ui/                     # Interface
│       ├── dashboard.py        # PySide6 Control Panel
│       └── window.py           # OpenCV Display Window wrapper
├── assets/
│   └── config/                 # Saved calibration data
├── requirements.txt
└── README.md

```

---

## 4. Component Deep Dive

### 4.1. The Main Engine (`main.py`)

This file orchestrates the application. It runs as a `QThread` to prevent the GUI from freezing during heavy image processing.

* **State Machine:** The engine switches behavior based on `self.mode`:
* `RUN`: Standard operation (Capture -> Process -> Sim -> Render).
* `CALIB_SEARCH`: Projects a checkerboard and searches the RGB stream for corners.
* `ROI`: Projects a black screen and displays raw depth to allow the user to draw the sandbox boundaries.


* **Signals & Slots:** Communicates with the `GeoBoxDashboard` using Qt Signals (thread-safe event passing).

### 4.2. Calibration & Geometry (`core/calibration.py`)

This is the mathematical core. Standard 2D Homography is insufficient because the Kinect's camera and projector are at different angles; as sand gets higher, the projection "slides" off the peak.

**The 2.5D Solution:**
We capture calibration points at two levels: **Base** (Low) and **Top** (High). We then solve a Linear Regression to find a matrix  that maps:

This allows the software to predict exactly where a pixel should land on the projector based on its 3D depth, eliminating "shadows" on mountains.

### 4.3. Depth Processor (`core/processor.py`)

Raw Kinect data is noisy and full of holes.

1. **Temporal Filter:** Uses `cv2.accumulateWeighted` to blend the current frame with previous frames (alpha ~0.5), smoothing out jitter.
2. **Shadow In-Painting:** The Kinect cannot see behind mountains (Parallax Occlusion). We detect these "zero" holes and use `cv2.inpaint` (Navier-Stokes method) to guess the missing terrain based on surrounding pixels.
3. **Contour Generation:**
* Upscales the height map 2x (Cubic Interpolation).
* Thresholds the map at regular intervals (e.g., 0.05, 0.10...).
* Extracts vector chains using `cv2.findContours`.
* Draws anti-aliased lines.



### 4.4. Fluid Simulation (`modules/fluid_sim.py`)

Implements the **Virtual Pipes / Shallow Water Model**.

* **Grid Optimization:** Runs on a downscaled grid () to maintain performance.
* **Logic:**
1. Calculate `Total Height = Sand Height + Water Depth`.
2. Compute flux (flow) to left/right/up/down neighbors based on gravity.
3. Limit flux so we don't move more water than exists in a cell.
4. Update water depth and render as a blue overlay.



### 4.5. The Dashboard (`ui/dashboard.py`)

A comprehensive control center built with **PySide6**.

* **Tabs:** Organized into Visuals, Fluids, and Calibration.
* **ROI Selector:** A transparent overlay widget (`ROISelector`) that sits on top of the video feed, allowing users to draw the simulation boundary with the mouse.
* **Video Widget:** A custom `QWidget` that paints the OpenCV frame directly, handling aspect ratio scaling without breaking the layout.

---

## 5. Development Workflow

### Adding a New Feature

1. **Logic:** Create the logic in `src/modules/`. Ensure it works with NumPy arrays (avoid Python `for` loops where possible).
2. **State:** Add state variables to `GeoBoxEngine` in `main.py` (e.g., `self.new_feature_enabled`).
3. **UI:** Add controls to `GeoBoxDashboard` in `ui/dashboard.py` and define a Signal (e.g., `new_feature_toggled`).
4. **Wiring:** Connect the UI Signal to the Engine Slot in `main.py`.

### Thread Safety Rules

* **Never** call GUI functions (e.g., `label.setText`) from `GeoBoxEngine`. It will crash the app. Use Signals (`self.signals.status_update.emit("Text")`).
* **Never** perform heavy computation (image processing) in `GeoBoxDashboard`. It will freeze the interface.

---

## 6. Troubleshooting & Common Issues

| Issue | Cause | Solution |
| --- | --- | --- |
| **Crash: `unpack expected 3, got 2**` | The Engine sent a Grayscale image (2D) to the Dashboard, which expected Color (3D). | Ensure `update_feed` checks `len(frame.shape)` before creating `QImage`. |
| **Blue Shadows on Sand** | The Kinect returns "Unknown Depth" (2047) or "Too Close" (0). | The `DepthProcessor` interprets these values as "Deep Floor". Ensure the In-Painting logic (Section 4.3) is active. |
| **"AttributeError: set_sea"** | A setter method is defined in the UI Signal connection but missing in the Engine class. | Check `main.py` and define the missing method in `GeoBoxEngine`. |
| **Laggy Fluid** | The physics grid is too large. | Reduce `physics_scale` in `fluid_sim.py` (Default: 0.25). |

---

## 7. Calibration Guide (User Manual)

1. **Start:** Launch app, go to **Calibration Tab**.
2. **ROI:** Click "Draw Sand Box". On the laptop screen, draw a box around the active sand area.
3. **Base:** Flatten the sand. Click "Capture Base". Wait for the checkerboard to be detected.
4. **Top:** Place a flat board on top of the box walls. Click "Capture Top".
5. **Save:** Click "Compute & Save". The sand should now align perfectly with the projection.

---

## 8. Dependencies

* **Python 3.8+**
* **libfreenect:** Low-level driver for Kinect v1.
* **NumPy:** Matrix math.
* **OpenCV (`opencv-python`):** Image processing.
* **PySide6:** GUI Framework.
* **Matplotlib:** For generating colormap LUTs.
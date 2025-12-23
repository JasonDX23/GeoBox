Original Layout

GeoBox/
├── src/                    # All source code
│   ├── main.py             # Entry point (GUI initialization)
│   ├── core/               # Fundamental logic (Internal APIs)
│   │   ├── kinect.py       # Libfreenect wrapper and stream handling
│   │   ├── calibration.py  # Homography and projection logic
│   │   └── processor.py    # Depth-to-height normalization
│   ├── modules/            # Extensible feature directory
│   │   ├── rain_sim.py     # Rain/Water physics
│   │   └── color_maps.py   # Gradient definitions
│   └── ui/                 # Interface layouts (PyQt/Tkinter)
├── assets/                 # Non-code resources
│   ├── shaders/            # GLSL scripts for rendering
│   └── config/             # YAML/JSON for calibration data
├── build/                  # Target for PyInstaller/EXE output
├── requirements.txt        # Dependency list
└── README.md
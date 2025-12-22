import sys
import os

# Ensure the 'src' directory is in the python path for easy imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from PySide6.QtWidgets import QApplication
from ui.main_window import ARSMainWindow

def main():
    # 1. Initialize the Qt Application
    app = QApplication(sys.argv)
    app.setStyle('Fusion') # Standardize look across Windows versions

    # 2. Instantiate the Main Window
    # This will automatically start the KinectWorker thread
    window = ARSMainWindow()
    window.show()

    # 3. Execute the Application loop
    try:
        sys.exit(app.exec())
    except SystemExit:
        print("Closing GeoBox ARS...")
        # Ensure threads stop correctly
        window.worker.stop()

if __name__ == "__main__":
    main()
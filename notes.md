PENDING: TODO:
- Add Sensor connection status near bottom left of the app
- add sea level offset
- option to display kinect depth view
- 

26/12/25 - Added ContourMatch Module
28/12/25 - Tried ProjectorKinect Calibration Module, needs to be implemented

## kinect.py
- gets kinect data in depth and rgb
- smoothens the data
- get colour frame (get_latest_rgb)

## processor.py
- contains TerrainProcessor
    - assigns colormap, base depth, contours
    - will slope calculation be needed?


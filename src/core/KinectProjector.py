import numpy as np
import cv2
import json

class KinectProjector:
    def __init__(self, proj_w=1024, proj_h=768):
        self.proj_res = (proj_w, proj_h)
        
        # Calibration Data Containers
        self.kinect_3d_pts = [] # List of (u, v, d)
        self.proj_2d_pts = []   # List of (x, y)
        
        # Kinect v1 Intrinsic Parameters (Typical values)
        # These convert pixels (u,v) + depth (d) into real-world (X,Y,Z)
        self.fx, self.fy = 580.0, 580.0
        self.cx, self.cy = 320.0, 240.0
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # The result: The Projection Matrix
        self.projection_matrix = None

    def add_point_pair(self, proj_pt, kinect_pt_uvd):
        """Adds a matched pair from the Calibration App."""
        self.proj_2d_pts.append(proj_pt)
        self.kinect_3d_pts.append(kinect_pt_uvd)

    def get_target_pts_for_pos(self, x_off, y_off, sq_sz=80, cols=6, rows=5):
        """Generates the 2D coordinates of the chessboard corners on the projector."""
        targets = []
        for y in range(1, rows + 1):
            for x in range(1, cols + 1):
                targets.append((x_off + x * sq_sz, y_off + y * sq_sz))
        return np.array(targets, dtype=np.float32)

    def solve_matrix(self):
        """
        Calculates the transformation matrix.
        Converts Kinect (u,v,d) -> World (X,Y,Z) -> Projector (x,y)
        """
        if len(self.kinect_3d_pts) < 10:
            return False

        # 1. Convert Kinect pixels + depth into 3D Camera Space (Object Points)
        obj_pts = []
        for u, v, d in self.kinect_3d_pts:
            z = float(d)
            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            obj_pts.append([x, y, z])
        
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(self.proj_2d_pts, dtype=np.float32)

        # 2. Use solvePnP to find the Projector's position and rotation
        # We assume the Projector acts like a camera in reverse
        success, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, self.camera_matrix, None)

        if success:
            # 3. Build a combined matrix for fast lookups
            rmat, _ = cv2.Rodrigues(rvec)
            # Create a 3x4 projection matrix
            self.projection_matrix = np.hstack((rmat, tvec))
            return True
        return False

    def project_point(self, u, v, d):
        """The 'Translator': Converts a Kinect pixel to a Projector pixel."""
        if self.projection_matrix is None or d <= 0:
            return None

        # Convert to 3D Camera Space
        z = float(d)
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        
        # Multiply by Projection Matrix
        world_pt = np.array([x, y, z, 1.0])
        # Result is in Projector-space coordinates
        proj_pt_h = self.camera_matrix @ (self.projection_matrix @ world_pt)
        
        # Perspective divide
        px = proj_pt_h[0] / proj_pt_h[2]
        py = proj_pt_h[1] / proj_pt_h[2]
        
        return int(px), int(py)

    def save_calibration(self, filename="calibration.json"):
        if self.projection_matrix is not None:
            data = {
                "projection_matrix": self.projection_matrix.tolist(),
                "camera_matrix": self.camera_matrix.tolist()
            }
            with open(filename, 'w') as f:
                json.dump(data, f)

    def load_calibration(self, filename="calibration.json"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.projection_matrix = np.array(data["projection_matrix"])
                self.camera_matrix = np.array(data["camera_matrix"])
            return True
        except:
            return False
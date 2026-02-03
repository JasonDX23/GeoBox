import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom

class ProjectorCalibrator:
    def __init__(self, width=1024, height=768):
        self.proj_res = (width, height) 
        self.kinect_res = (640, 480)
        self.coeffs = np.zeros(11, dtype=np.float64)
        self.calibrated = False
        self.collected_in, self.collected_out = [], []
        self.roi = [0, 0, 640, 480]
        self._last_offset = (0, 0)
        
        # Precompute grid at FULL PROJECTOR resolution
        self.grid_u_proj, self.grid_v_proj = np.meshgrid(
            np.arange(self.proj_res[0]), np.arange(self.proj_res[1])
        )
        self.board_dims, self.sq_size = (4, 3), 60 

    @property
    def matrix(self): return self.coeffs

    def set_config(self, roi, matrix):
        self.roi = roi
        if matrix is not None:
            m_np = np.array(matrix).flatten()
            if len(m_np) == 11:
                self.coeffs = m_np
                self.calibrated = True

    def reset_collection(self):
        self.collected_in, self.collected_out, self.calibrated = [], [], False

    def set_config(self, roi, matrix_list):
        self.roi = roi
        mat = np.array(matrix_list, dtype=np.float32)
        # Force resize to (2, 6) if an old config is loaded
        if mat.shape != (2, 6):
            self.matrix = np.zeros((2, 6), dtype=np.float32)
            self.matrix[0, 0], self.matrix[1, 1] = 1.0, 1.0
        else:
            self.matrix = mat

    def capture_frame(self, color, depth, step):
        viz = color.copy()
        ret, corners = cv2.findChessboardCorners(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY), (4,3), None)
        if ret:
            sx, sy = self._last_offset
            for i in range(3):
                for j in range(4):
                    tx, ty = sx + (j+1)*60, sy + (i+1)*60
                    kx, ky = corners[i*4+j][0]
                    kz = depth[min(int(ky), 479), min(int(kx), 639)]
                    if kz > 0:
                        self.collected_in.append([kx, ky, kz])
                        self.collected_out.append([tx, ty])
            cv2.drawChessboardCorners(viz, (4,3), corners, ret)
            return True, viz
        return False, viz

    def compute(self):
        n = len(self.collected_in)
        if n < 6: return False
        A, y = np.zeros((n*2, 11)), np.zeros((n*2, 1))
        for i in range(n):
            pk, pp = self.collected_in[i], self.collected_out[i]
            A[2*i, 0:4], y[2*i] = [pk[0], pk[1], pk[2], 1], pp[0]
            A[2*i, 8:11] = [-pk[0]*pp[0], -pk[1]*pp[0], -pk[2]*pp[0]]
            A[2*i+1, 4:8], y[2*i+1] = [pk[0], pk[1], pk[2], 1], pp[1]
            A[2*i+1, 8:11] = [-pk[0]*pp[1], -pk[1]*pp[1], -pk[2]*pp[1]]
        q, r = np.linalg.qr(A)
        self.coeffs = np.linalg.solve(r, q.T @ y).flatten()
        self.calibrated = True
        self.save_xml_files()
        return True

    def warp(self, depth):
        if not self.calibrated: return np.zeros((self.proj_res[1], self.proj_res[0]), dtype=np.uint16)
        z = cv2.resize(depth, self.proj_res, interpolation=cv2.INTER_NEAREST).astype(float)
        u, v = self.grid_u_proj, self.grid_v_proj
        denom = np.maximum(self.coeffs[8]*u + self.coeffs[9]*v + self.coeffs[10]*z + 1.0, 1e-6)
        mx = (self.coeffs[0]*u + self.coeffs[1]*v + self.coeffs[2]*z + self.coeffs[3]) / denom
        my = (self.coeffs[4]*u + self.coeffs[5]*v + self.coeffs[6]*z + self.coeffs[7]) / denom
        return cv2.remap(depth, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR)

    def save_xml_files(self):
        root = ET.Element("CALIBRATION")
        res = ET.SubElement(root, "RESOLUTIONS")
        ET.SubElement(res, "PROJECTOR").text = f"{self.proj_res[0]}, {self.proj_res[1]}"
        coeffs = ET.SubElement(root, "COEFFICIENTS")
        for i, v in enumerate(self.coeffs): ET.SubElement(coeffs, f"COEFF{i}").text = f"{v:g}"
        with open("calibration.xml", "w") as f: f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="\t"))
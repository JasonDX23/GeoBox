# Module has to be worked on

# Inside src/modules/rain_sim.py
import cv2
import numpy as np

def calculate_slopes(depth_map):
    # Calculate horizontal and vertical gradients
    dz_dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    dz_dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)
    return dz_dx, dz_dy

class RainSimulation:
    def __init__(self, count=300):
        # Initialize particles randomly across the 640x480 space
        self.particles = np.random.rand(count, 2) 
        self.particles[:, 0] *= 639 # X bounds
        self.particles[:, 1] *= 479 # Y bounds

    def update(self, dz_dx, dz_dy):
        for i in range(len(self.particles)):
            # Convert float positions to integer indices
            px = int(self.particles[i][0])
            py = int(self.particles[i][1])

            # Check bounds BEFORE indexing to avoid IndexError
            if 0 <= px < 639 and 0 <= py < 479:
                # Update velocity: Move in the direction of the gradient
                # dz_dx[py, px] follows [row, col] -> [y, x]
                self.particles[i][0] -= dz_dx[py, px] * 0.1
                self.particles[i][1] -= dz_dy[py, px] * 0.1
            else:
                # Reset particle if it falls off the edge of the sandbox
                self.particles[i] = [np.random.uniform(0, 639), np.random.uniform(0, 479)]
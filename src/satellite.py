import math
import numpy as np
from scipy.spatial.transform import Rotation

from .orbit import Orbit
from .constants import earth_radius

class Satellite:
    def __init__(self, mass: float, orbit: Orbit):
        self.mass = mass
        self.orbit = orbit
        self.state = Orbit.Coes2State(orbit.coes) # state = [rx, ry, rz, ax, ay, az, t]

    # Transform Earth-Centered Inertial to Earth-Centered Earth-Fixed
    @staticmethod
    def Eci2Ecef(state: np.ndarray[float], t: float) -> np.ndarray[float]:
        omega = 0.261799387799149  # radians/hour
        # theta = Cal2Gmst()
        theta = float((omega * t / 60 / 60) % (2 * math.pi))
        R = Rotation.from_euler("Z", theta, degrees=False)
        return np.concatenate((state[:3] @ R.as_matrix(), state[3:6] @ R.as_matrix()))

    # Transform Earth-Centered Earth-Fixed to Geocentric coordinates
    @staticmethod
    def Ecef2Geoc(state: np.ndarray[float], r: float):
        geoc = np.zeros(3)

        geoc[0] = math.degrees(math.atan2(state[1], state[0]))  # longitude
        geoc[1] = math.degrees(math.asin(state[2] / np.linalg.norm(state)))  # latitude
        geoc[2] = np.linalg.norm(state) - r  # altitude

        return geoc
    
    def get_state_eci(self) -> np.ndarray[float]:
        return self.state
    
    def get_state_ecef(self, t: float) -> np.ndarray[float]:
        return self.Eci2Ecef(self.state, t)

    def get_state_geoc(self, t: float) -> np.ndarray[float]:
        return self.Ecef2Geoc(self.get_state_ecef(t), earth_radius)
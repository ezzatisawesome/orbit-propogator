import math
import numpy as np
from scipy.spatial.transform import Rotation

from .orbit import Orbit
from .constants import earth_radius, earth_mu, sun_mu
from .utils import get_gmst_from_epoch

class Satellite:
    def __init__(self, mass: float, orbit: Orbit, options: dict = {}):
        self.mass = mass
        self.orbit = orbit
        self.options = options
        self.mu = sun_mu if ('sun' in options and options['sun']) else earth_mu
        self.state = Orbit.Coes2State(orbit.coes, self.mu) # state = [rx, ry, rz, ax, ay, az, t]

    # Transform Earth-Centered Inertial to Earth-Centered Earth-Fixed
    @staticmethod
    def Eci2Ecef(state: np.ndarray[float], t: float) -> np.ndarray[float]:
        # omega = 0.261799387799149  # radians/hour
        # theta = Cal2Gmst()
        # theta = float((omega * t / 60 / 60) % (2 * math.pi))
        theta = get_gmst_from_epoch(t)
        R = Rotation.from_euler("Z", theta, degrees=False)
        return state @ R.as_matrix()

    # Transform Earth-Centered Earth-Fixed to Geocentric coordinates
    @staticmethod
    def Ecef2Geoc(state: np.ndarray[float], r: float):
        geoc = np.zeros(3)
        geoc[0] = math.degrees(math.atan2(state[1], state[0]))  # longitude
        geoc[1] = math.degrees(math.asin(state[2] / np.linalg.norm(state)))  # latitude
        geoc[2] = np.linalg.norm(state) - r  # altitude
        return geoc
    
    '''
    @return earth-centered inertial coordinates [x, y, z, vx, vy, vz]
    '''
    def get_state_eci(self) -> np.ndarray[float]:
        return self.state
    
    '''
    @param t: time in seconds since epoch
    @return earth-centered earth-fixed coordinates [x, y, z]
    '''
    def get_state_ecef(self, t: float) -> np.ndarray[float]:
        state = self.state[:3]
        return self.Eci2Ecef(state, t)

    '''
    @param t: time in seconds since epoch
    @return geocentric coordinates [longitude, latitude, altitude]
    '''
    def get_state_geoc(self, t: float) -> np.ndarray[float]:
        return self.Ecef2Geoc(self.get_state_ecef(t), earth_radius)
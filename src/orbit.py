import math
import numpy as np
from typing import Tuple
from scipy.spatial.transform import Rotation

from .constants import earth_mu

class Orbit:
    def __init__(self, coes: Tuple[float, float, float, float, float, float]):
        self.coes = coes
    
    # Convert Classical Orbital Elements to State Vector
    @staticmethod
    def Coes2State(coes: Tuple[float, float, float, float, float, float]) -> np.ndarray[float]:
        sma, ecc, inc, ta, aop, raan = coes

        # calculate orbital angular momentum of satellite
        h = math.sqrt(earth_mu * (sma * (1 - ecc**2)))
        # h = 1.6041e13

        cos_ta = math.cos(math.radians(ta))
        sin_ta = math.sin(math.radians(ta))

        print(h**2)

        r_w = h ** 2 / earth_mu / (1 + ecc * cos_ta) * \
            np.array((cos_ta, sin_ta, 0))
        v_w = earth_mu / h * np.array((-sin_ta, ecc + cos_ta, 0))

        # rotate to inertian frame
        R = Rotation.from_euler("ZXZ", [-aop, -inc, -raan], degrees=True)
        r_rot = r_w @ R.as_matrix()
        v_rot = v_w @ R.as_matrix()

        print(v_rot)

        return np.concatenate((r_rot, v_rot))
    
    @staticmethod
    def State2Coes(state: np.ndarray[float]) -> Tuple[float, float, float, float, float, float]:
        r_vec = state[:3]
        v_vec = state[3:]

        # Position and velocity magnitudes
        r = np.linalg.norm(r_vec)
        v = np.linalg.norm(v_vec)
        v_r = np.dot(r_vec / r, v_vec)
        v_p = np.sqrt(v ** 2 - v_r ** 2)

        # Orbital angular momentum
        h_vec = np.cross(r_vec, v_vec)
        h = np.linalg.norm(h_vec)
        sma = h

        # Inclination
        inc = np.arccos(h_vec[2] / h)

        # RAAN
        K = np.array((0, 0, 1))
        N_vec = np.cross(K, h_vec)
        N = np.linalg.norm(N_vec)
        raan = 2 * np.pi - np.arccos(N_vec[0] / N)

        # Eccentricity
        e_vec = np.cross(v_vec, h_vec) / earth_mu - r_vec / r
        ecc = np.linalg.norm(e_vec)

        # AOP
        aop = 2 * np.pi - np.arccos(np.dot(N_vec, e_vec) / (N * ecc))

        # True anomaly
        ta = np.arccos(np.dot(r_vec / r, e_vec / ecc))


        return np.array((sma, ecc, inc, ta, aop, raan))
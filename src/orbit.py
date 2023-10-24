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

        # calculate velocity of satellite
        h = 7641.8 * 7.22222  # km^2/s

        cos_ta = math.cos(math.radians(ta))
        sin_ta = math.sin(math.radians(ta))

        r_w = h ** 2 / earth_mu / (1 + ecc * cos_ta) * \
            np.array((cos_ta, sin_ta, 0))
        v_w = earth_mu / h * np.array((-sin_ta, ecc + cos_ta, 0))

        # rotate to inertian frame
        R = Rotation.from_euler("ZXZ", [-aop, -inc, -raan], degrees=True)
        r_rot = r_w @ R.as_matrix()
        v_rot = v_w @ R.as_matrix()

        return np.concatenate((r_rot, v_rot))
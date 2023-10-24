import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

from .satellite import Satellite
from .plot import plot_groundtracks
from .constants import earth_mu, earth_radius, earth_j2

class Propagate:
    def __init__(self, satellite: Satellite, tspan: int, dt: int):
        self.satellite = satellite
        self.tspan = tspan
        self.dt = dt
        self.steps = int(tspan / dt)

    # state = [rx, ry, rz, vx, vy, vz]
    # Acceleration due to Gravity
    @staticmethod
    def TwoBodyODE(state: np.ndarray[float]) -> np.ndarray[float]:
        # Get the position vector from state
        r_vector = state[:3]

        # Calculate the acceleration vector
        a_vector = (-earth_mu / np.linalg.norm(r_vector) ** 3) * r_vector

        # Return the derivative of the state
        return np.array([a_vector[0], a_vector[1], a_vector[2]])

    # state = [rx, ry, rz, vx, vy, vz]
    # Acceleration due to J2 Perturbation
    @staticmethod
    def J2Pert(state: np.ndarray[float]) -> np.ndarray[float]:
        r_norm = np.linalg.norm(state[:3])
        r_squared = r_norm ** 2
        z_squared = state[2] ** 2

        p = (3 * earth_j2 * earth_mu * (earth_radius ** 2)) / \
            (2 * (r_squared ** 2))

        ax = ((5 * z_squared / r_squared) - 1) * (state[0] / r_norm)
        ay = ((5 * z_squared / r_squared) - 1) * (state[1] / r_norm)
        az = ((5 * z_squared / r_squared) - 3) * (state[2] / r_norm)

        return np.array([ax, ay, az]) * p

    # state = [rx, ry, rz, vx, vy, vz]
    # Differential equation for the state vector
    @staticmethod
    def DiffEqn(state: np.ndarray[float]) -> np.ndarray[float]:
        rx, ry, rz, vx, vy, vz = state

        # state_dot = TwoBodyODE(state)
        state_dot = np.zeros(6)

        # Newton's Universal Law of Gravitation
        a = Propagate.TwoBodyODE(state)
        a += Propagate.J2Pert(state)

        # Return the derivative of the state
        state_dot[:3] = [vx, vy, vz]
        state_dot[3:6] = a

        return state_dot

    # Runge Kutta solver for the differential equation
    @staticmethod
    def RK4(
        fn: Callable[[np.ndarray[float]], np.ndarray[float]],
        y: float,
        h: float
    ):
        k1 = fn(y)
        k2 = fn(y + 0.5 * k1 * h)
        k3 = fn(y + 0.5 * k2 * h)
        k4 = fn(y + k3 * h)

        return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    
    # Propagate the state vector based provided inputs
    def propagate(self):
        states = np.zeros((self.steps, 6))
        states[0] = self.satellite.state

        statesGeoc = np.zeros((self.steps, 3))
        statesGeoc[0] = self.satellite.get_state_geoc(self.dt)

        for step in range(self.steps - 1):
            states[step + 1] = Propagate.RK4(self.DiffEqn, states[step], self.dt)
            statesGeoc[step + 1] = self.satellite.get_state_geoc(step * self.dt)
            self.satellite.state = states[step + 1]

        plot_groundtracks(statesGeoc)
        plt.show()
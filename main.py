import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple
from scipy.spatial.transform import Rotation

from coastlines import plot_groundtracks
from constants import earth_mu, earth_radius, earth_j2


def TwoBodyODE(state: np.ndarray[float]) -> np.ndarray[float]:
    # Get the position vector from state
    r_vector = state[:3]

    # Calculate the acceleration vector
    a_vector = (-earth_mu / np.linalg.norm(r_vector) ** 3) * r_vector

    # Return the derivative of the state
    return np.array([a_vector[0], a_vector[1], a_vector[2]])


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


def DiffEqn(t, state: np.ndarray[float]) -> np.ndarray[float]:
    rx, ry, rz, vx, vy, vz = state
    # Get the position vector from state
    r_vector = np.array([rx, ry, rz])

    # state_dot = TwoBodyODE(state)
    state_dot = np.zeros(6)

    # Newton's Universal Law of Gravitation
    a = TwoBodyODE(state)
    a += J2Pert(state)

    # Return the derivative of the state
    state_dot[:3] = [vx, vy, vz]
    state_dot[3:6] = a

    return state_dot


def RK4(
    fn: Callable[[np.ndarray[float]], np.ndarray[float]],
    t: float,
    y: float,
    h: float
):
    print(t)
    k1 = fn(t, y)
    k2 = fn(t, y + 0.5 * k1 * h)
    k3 = fn(t, y + 0.5 * k2 * h)
    k4 = fn(t, y + k3 * h)

    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


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

import math

def Cal2Gmst(Y1, M1, D1, D):
    # Compute modified month and year
    if M1 <= 2:
        Y2 = Y1 - 1
        M2 = M1 + 12
    else:
        Y2 = Y1
        M2 = M1

    B = Y1 / 400 - Y1 / 100 + Y1 / 4

    # Decimal days
    D2 = D1 + D

    # Modified Julian Date
    MJD = 365 * Y2 - 679004 + int(B) + int(30.6001 * (M2 + 1)) + D2
    d = MJD - 51544.5

    # GMST in degrees
    GMST = 280.4606 + 360.9856473 * d
    GMST = math.radians(GMST) # Convert to radians
    GMST = GMST % (2 * math.pi) # Ensure GMST is in the range [0, 2 * pi)

    return GMST



def Eci2Ecef(t, state: np.ndarray[float]) -> np.ndarray[float]:
    omega = 0.261799387799149  # radians/hour
    # theta = Cal2Gmst()
    theta = float((omega * t / 60 / 60) % (2 * math.pi))

    rotation_matrix = np.array([
        [math.cos(theta), -math.sin(theta), 0],
        [math.sin(theta), math.cos(theta), 0],
        [0, 0, 1]
    ])

    return state @ rotation_matrix

def Ecef2Geoc(state, r):
    geoc = np.zeros(3)

    geoc[0] = math.degrees(math.asin(state[2] / np.linalg.norm(state)))
    geoc[1] = math.degrees(math.atan2(state[1], state[0]))
    geoc[2] = np.linalg.norm(state) - r

    return geoc


def plot_orbits(rs, args):
    _args = {
        'figsize': (10, 8),
        'labels': [''] * len(rs),
        'colors': ['m', 'c', 'r', 'C3'],
        'traj_lws': 3,
        'dist_unit': 'km',
        'groundtracks': False,
        'cb_radius': 6378.0,
        'cb_SOI': None,
        'cb_SOI_color': 'c',
        'cb_SOI_alpha': 0.7,
        'cb_axes': True,
        'cb_axes_mag': 2,
        'cb_cmap': 'Blues',
        'cb_axes_color': 'w',
        'axes_mag': 0.8,
        'axes_custom': None,
        'title': 'Trajectories',
        'legend': True,
        'axes_no_fill': True,
        'hide_axes': False,
        'azimuth': False,
        'elevation': False,
        'show': False,
        'filename': False,
        'dpi': 300
    }
    for key in args.keys():
        _args[key] = args[key]

    fig = plt.figure(figsize=_args['figsize'])
    ax = fig.add_subplot(111, projection='3d')

    max_val = 0
    n = 0

    for r in rs:
        ax.plot(r[:, 0], r[:, 1], r[:, 2],
                color=_args['colors'][n], label=_args['labels'][n],
                zorder=10, linewidth=_args['traj_lws'])
        ax.plot([r[0, 0]], [r[0, 1]], [r[0, 2]], 'o',
                color=_args['colors'][n])

        if _args['groundtracks']:
            rg = r[:] / np.linalg.norm(r, axis=1).reshape((r.shape[0], 1))
            rg *= _args['cb_radius']

            ax.plot(rg[:, 0], rg[:, 1], rg[:, 2], cs[n], zorder=10)
            ax.plot([rg[0, 0]], [rg[0, 1]], [rg[0, 2]], cs[n] + 'o', zorder=10)

        max_val = max([r.max(), max_val])
        n += 1

    _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:20j]
    _x = _args['cb_radius'] * np.cos(_u) * np.sin(_v)
    _y = _args['cb_radius'] * np.sin(_u) * np.sin(_v)
    _z = _args['cb_radius'] * np.cos(_v)
    ax.plot_surface(_x, _y, _z, cmap=_args['cb_cmap'], zorder=1)

    if _args['cb_axes']:
        l = _args['cb_radius'] * _args['cb_axes_mag']
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
        ax.quiver(x, y, z, u, v, w, color=_args['cb_axes_color'])

    xlabel = 'X (%s)' % _args['dist_unit']
    ylabel = 'Y (%s)' % _args['dist_unit']
    zlabel = 'Z (%s)' % _args['dist_unit']

    if _args['axes_custom'] is not None:
        max_val = _args['axes_custom']
    else:
        max_val *= _args['axes_mag']

    ax.set_xlim([-max_val, max_val])
    ax.set_ylim([-max_val, max_val])
    ax.set_zlim([-max_val, max_val])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect('auto')

    if _args['azimuth'] is not False:
        ax.view_init(elev=_args['elevation'],
                     azim=_args['azimuth'])

    # if _args['axes_no_fill']:
    #     ax.w_xaxis.pane.fill = False
    #     ax.w_yaxis.pane.fill = False
    #     ax.w_zaxis.pane.fill = False

    if _args['hide_axes']:
        ax.set_axis_off()

    if _args['legend']:
        plt.legend()

    if _args['filename']:
        plt.savefig(_args['filename'], dpi=_args['dpi'])
        print('Saved', _args['filename'])

    if _args['show']:
        plt.show()

    plt.close()


if __name__ == '__main__':
    # coes = sma, ecc, inc, ta, aop, raan
    coes = [7641.80, 0.0, 98.94, 0, 0, 0]
    # coes = [7500, 0.0, 25.0, 0.0, 0.0, 0.0]

    # Convert orbital elements to state vector
    statei = Coes2State(coes)
    print(statei)

    tspan = 60.0 * 100  # seconds
    dt = 100.0  # seconds
    steps = int(tspan / dt)
    ets = np.zeros((steps, 1))
    states = np.zeros((steps, 6))
    states[0] = statei

    statesECEF = np.zeros((steps, 3))
    statesECEF[0] = Eci2Ecef(ets[0], states[0][:3])

    statesGeoc = np.zeros((steps, 3))
    statesGeoc[0] = Ecef2Geoc(statesECEF[0], earth_radius)

    for step in range(steps - 1):
        states[step + 1] = RK4(
            DiffEqn, ets[step], states[step], dt)

        statesECEF[step + 1] = Eci2Ecef(dt*step, states[step][:3])
        statesGeoc[step + 1] = Ecef2Geoc(statesECEF[step], earth_radius)

    # plot the groundtrack
    plot_groundtracks(statesGeoc)
    plt.show()

    # plot_orbits([states], {'show': True})

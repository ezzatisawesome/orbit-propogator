from typing import Callable
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation

from constants import earth_mu, earth_radius, earth_j2


def TwoBodyODE(state: ndarray[float]) -> ndarray[float]:
    # Get the position vector from state
    r_vector = state[:3]

    # Newton's Universal Law of Gravitation
    a_vector = (-earth_mu / np.linalg.norm(r_vector) ** 3) * r_vector

    # Return the derivative of the state
    return np.array([a_vector[0], a_vector[1], a_vector[2]])

def J2Pert(state: ndarray[float]) -> ndarray[float]:
    rNorm = np.linalg.norm(state[:3])
    rSquared = rNorm ** 2
    zSquared = state[2] ** 2

    p = (3 * earth_j2 * earth_mu * (earth_radius ** 2)) / \
        (2 * (rSquared ** 2))

    ax = ((5 * zSquared / rSquared) - 1) * (state[0] / rNorm)
    ay = ((5 * zSquared / rSquared) - 1) * (state[1] / rNorm)
    az = ((5 * zSquared / rSquared) - 3) * (state[2] / rNorm)

    return np.array([ax, ay, az]) * p

def Coes2State(coes) -> ndarray[float]:
    sma, ecc, inc, ta, aop, raan = coes

    # calculate velocity of satellite
    h = 7641.8 * 7.22222 # km^2/s

    cos_ta = np.cos(math.radians(ta))
    sin_ta = np.sin(math.radians(ta))

    r_w = h ** 2 / earth_mu / (1 + ecc * np.cos(ta)) * np.array((np.cos(ta), np.sin(ta), 0))
    v_w = earth_mu / h * np.array((-np.sin(ta), ecc + np.cos(ta), 0))


    # r_w = h ** 2 / earth_mu / (1 + ecc * cos_ta) * np.array((cos_ta, sin_ta, 0))
    # v_w = earth_mu / h * np.array((-sin_ta, ecc + cos_ta, 0))

    # rotate to perifocal frame
    R = Rotation.from_euler("ZXZ", [-aop, -inc, -raan], degrees=True)
    r_rot = r_w @ R.as_matrix()
    v_rot = v_w @ R.as_matrix()

    return np.concatenate((r_rot, v_rot))

def DiffEqn(t, state):
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
    fn: Callable[[ndarray[float]], ndarray[float]],
    t: float,
    y: float,
    h: float
):

    k1 = fn(t, y)
    k2 = fn(t, y + 0.5 * k1 * h)
    k3 = fn(t, y + 0.5 * k2 * h)
    k4 = fn(t, y + k3 * h)

    return y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

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
    coes = [ 7641.80, 0.0, 98.94, 0, 0, 45 ]
    # coes = [ 7641.80, 0.948, 124.05, 159.61, 303.09, 190.62 ]

    # Convert orbital elements to state vector
    statei = Coes2State(coes)
    print(statei)

    tspan = 1000 * 60.0 * 10            # seconds
    dt = 100.0                            # seconds
    steps = int(tspan / dt)
    ets = np.zeros((steps, 1))
    states = np.zeros((steps, 6))
    states[0] = statei

    for step in range(steps - 1):
        states[step + 1] = RK4(
            DiffEqn, ets[step], states[step], dt)

    plot_orbits([states], {'show': True})

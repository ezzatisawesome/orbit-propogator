from typing import Callable
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
import math

from constants import earth_mu, earth_radius, earth_j2


def TwoBodyODE(state: ndarray[float]) -> ndarray[float]:
    # Get the position vector from state
    r_vector = state[:3]

    # Newton's Universal Law of Gravitation
    a_vector = (-earth_mu / np.linalg.norm(r_vector) ** 3) * r_vector

    # Return the derivative of the state
    return np.array([state[3], state[4], state[5], a_vector[0], a_vector[1], a_vector[2]])


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


# def calc_J2(self, et, state):
#     z2 = state[2] ** 2
#     norm_r = nt.norm(state[:3])
#     r2 = norm_r ** 2
#     tx = state[0] / norm_r * (5 * z2 / r2 - 1)
#     ty = state[1] / norm_r * (5 * z2 / r2 - 1)
#     tz = state[2] / norm_r * (5 * z2 / r2 - 3)
#     return 1.5 * self.cb['J2'] * self.cb['mu'] *\
#         self.cb['radius'] ** 2 \
#         / r2 ** 2 * np.array([tx, ty, tz])

def J2(state):
    rNorm = np.linalg.norm(state[:3])
    rSquared = rNorm ** 2
    zSquared = state[2] ** 2

    p = (3 * earth_j2 * earth_mu * (earth_radius ** 2)) / \
        (2 * (rSquared ** 2))

    ax = ((5 * zSquared / rSquared) - 1) * (state[0] / rNorm)
    ay = ((5 * zSquared / rSquared) - 1) * (state[1] / rNorm)
    az = ((5 * zSquared / rSquared) - 3) * (state[2] / rNorm)

    return np.array([ax, ay, az]) * p


def DiffEqn(t, state):
    rx, ry, rz, vx, vy, vz = state
    # Get the position vector from state
    r_vector = np.array([rx, ry, rz])

    # state_dot = TwoBodyODE(state)
    state_dot = np.zeros(6)

    # Newton's Universal Law of Gravitation
    a = (-earth_mu / np.linalg.norm(r_vector) ** 3) * r_vector
    a += J2(state)

    state_dot[:3] = [vx, vy, vz]
    state_dot[3:6] = a
    
    return state_dot

# function coes2state( coes ) {
# 	let [ sma, ecc, inc, ta, aop, raan ] = coes;
# 	sta     = Math.sin( ta );
# 	cta     = Math.cos( ta );
# 	p       = sma * ( 1 - Math.pow( ecc, 2 ) );
# 	r_norm  = p / ( 1 + ecc * cta );
# 	r_perif = math.multiply( math.matrix( [ cta, sta, 0 ] ), r_norm );
# 	v_perif = math.multiply( math.matrix( [ -sta, ecc + cta, 0 ] ),
# 					 Math.sqrt( CB[ 'mu' ] / p ) );
# 	matrix  = perif2eci( raan, aop, inc );
# 	r_ECI   = math.multiply( matrix, r_perif );
# 	v_ECI   = math.multiply( matrix, v_perif );
# 	return math.concat( r_ECI, v_ECI ).valueOf();

# function perif2eci( raan, aop, inc ) {
# 	matrix = math.multiply( Cz( raan ), Cx( inc ) );
# 	return math.multiply( matrix, Cz( aop ) );
# }

# function Cx( a ) {
# 	sa = Math.sin( a );
# 	ca = Math.cos( a );
# 	return math.matrix( [ 
# 		[ 1,  0,   0 ],
# 		[ 0, ca, -sa ],
# 		[ 0, sa,  ca ]
# 	] )
# }

# function Cy( a ) {
# 	sa = Math.sin( a );
# 	ca = Math.cos( a );
# 	return math.matrix( [
# 		[  ca, 0, sa ],
# 		[   0, 1,  0 ],
# 		[ -sa, 0, ca ]
# 	] )
# }

# function Cz( a ) {
# 	sa = Math.sin( a );
# 	ca = Math.cos( a );
# 	return math.matrix( [ 
# 		[ ca, -sa, 0 ],
# 		[ sa,  ca, 0 ],
# 		[  0,   0, 1 ]
# 	] )
# }

def Cx(a):
    sa = math.sin(math.radians(a))
    ca = math.cos(math.radians(a))
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ])

def Cy(a):
    sa = math.sin(math.radians(a))
    ca = math.cos(math.radians(a))
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ])

def Cz(a):
    sa = math.sin(math.radians(a))
    ca = math.cos(math.radians(a))
    return np.array([
        [ca, -sa, 0],
        [sa, ca, 0],
        [0, 0, 1]
    ])

def perif2eci(raan, aop, inc):
    matrix = np.matmul(Cz(raan), Cx(inc))
    return np.matmul(matrix, Cz(aop))

def coes2state( coes ):
    sma, ecc, inc, ta, aop, raan = coes
    sta = math.sin(math.radians(ta))
    cta = math.cos(math.radians(ta))
    p = sma * (1 - ecc ** 2)
    r_norm = p / (1 + ecc * cta)
    r_perif = np.array([cta, sta, 0]) * r_norm
    v_perif = np.array([-sta, ecc + cta, 0]) * math.sqrt(earth_mu / p)
    matrix = perif2eci(raan, aop, inc)

    r_ECI = np.matmul(matrix, r_perif)

    return r_ECI


def coes2state2( coes ):
    sma, ecc, inc, ta, aop, raan = coes

    
    # calculate velocity of satellite
    h = 7641.8 * 7.22222

    r_w = h ** 2 / earth_mu / (1 + ecc * np.cos(math.radians(ta))) * np.array((np.cos(math.radians(ta)), np.sin(math.radians(ta)), 0))
    v_w = earth_mu / h * np.array((-np.sin(math.radians(ta)), ecc + np.cos(math.radians(ta)), 0))

    #% Direction cosine matrix
    arr1 = np.array([
        [ np.cos(math.radians(aop)), np.sin(math.radians(aop)), 0 ],
        [ -np.sin(math.radians(aop)), np.cos(math.radians(aop)), 0 ],
        [ 0, 0, 1 ]
    ])
    arr2 = np.array([
        [ 1, 0, 0 ],
        [ 0, np.cos(math.radians(inc)), np.sin(math.radians(inc)) ],
        [ 0, -np.sin(math.radians(inc)), np.cos(math.radians(inc)) ]
    ])
    arr3 = np.array([
        [ np.cos(math.radians(raan)), np.sin(math.radians(raan)), 0 ],
        [ -np.sin(math.radians(raan)), np.cos(math.radians(raan)), 0 ],
        [ 0, 0, 1 ]
    ])

    QXx = np.matmul(np.matmul(arr3, arr2), arr1)

    # Transformation Matrix
    QxX = np.linalg.inv(QXx)

    # Geocentric equatorial position vector R
    R = np.matmul(QxX, r_w)

    # Geocentric equatorial velocity vector V
    V = np.matmul(QxX, v_w)

    # print(r_w)
    # print(v_w)
    # print(QXx)
    print(R)
    print(V)

    return np.concatenate((R, V))



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

#     Vo: does not apply 
# a = 7641.80 km
# e = 0
# dΩ/dt = 0.1991e-6 rad/s
# i = 98.94°

    coes = [ 7641.80, 0.0, 98.94, 0, 0, 45 ]




    # r0_norm = earth_radius + 450.0             # km
    r0_norm = 4992.02759388496
    v0_norm = (earth_mu / r0_norm) ** 0.5    # km / s
    # statei = [r0_norm, 0, 0, 0, 0, v0_norm]
    statei = coes2state2(coes)
    # statei = [4992.02759388496, 5434.057309954339, -1986.9043494901769, 2.109506463693961, 0.5784426780115759, 6.882065129398879]
    tspan = 60 * 60.0 * 24 * 180              # seconds
    dt = 100.0                            # seconds
    steps = int(tspan / dt)
    ets = np.zeros((steps, 1))
    states = np.zeros((steps, 6))
    states[0] = statei

    for step in range(steps - 1):
        states[step + 1] = RK4(
            DiffEqn, ets[step], states[step], dt)

    plot_orbits([states], {'show': True})

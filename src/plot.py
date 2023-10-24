import os
import matplotlib.pyplot as plt
import numpy as np

COASTLINES_COORDINATES_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('..', 'data', 'coastlines.csv')
)

EARTH_SURFACE_IMAGE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    os.path.join('..', 'data', 'earth_surface.png')
)

def plot_groundtracks(geocentric_states: np.ndarray[float]):
    plt.figure(figsize=(16, 8))
    coast_cords = np.genfromtxt(COASTLINES_COORDINATES_FILE, delimiter=',')
    plt.plot(coast_cords[:, 0], coast_cords[:, 1], 'mo', markersize=0.3)

    plt.plot(
        geocentric_states[:, 0],
        geocentric_states[:, 1],
        color='red',
        marker='o',
        linestyle='none',
    )

    plt.imshow(
        plt.imread(EARTH_SURFACE_IMAGE),
        extent=[-180, 180, -90, 90])

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